#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
import multiprocessing
from datetime import datetime
import numpy as np
import gymnasium as gym

# Suppress gymnasium deprecation warning for action_masks
warnings.filterwarnings('ignore', message='.*env.action_masks.*', category=UserWarning)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the wrapper from env.py (Crucial for Mac pickling)
from env import SkyBitesEnv, DroneActionMaskWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

def make_env(orders_csv, restaurants_json, pads_json, num_drones, rank=0, seed=0, is_eval=False,
             auto_chain_orders=False, order_scale_factor=1):
    """
    Utility to create the env. 
    Must be a standalone function (not nested) for best Mac compatibility.
    
    Args:
        orders_csv: Path to CSV file (only used for eval, None for training)
        restaurants_json: Path to restaurants config JSON
        pads_json: Path to delivery pads config JSON
        num_drones: Number of drones
        rank: Environment rank (for parallelization)
        seed: Base seed
        is_eval: If True, use static CSV. If False, use dynamic generation.
        auto_chain_orders: If True, drones automatically route restaurant->dropoff->base
        order_scale_factor: Scale factor for orders (1 = all orders, 20 = 1/20 of orders)
    """
    def _init():
        env = SkyBitesEnv(
            orders_csv=orders_csv if is_eval else None,
            restaurants_json=restaurants_json,
            pads_json=pads_json,
            num_drones=num_drones,
            dynamic_generation=(not is_eval),
            auto_chain_orders=auto_chain_orders,
            order_scale_factor=order_scale_factor
        )
        env = DroneActionMaskWrapper(env)
        env = Monitor(env, filename=None)
        # Reset must be called before any attribute access for multiprocessing
        obs, info = env.reset(seed=seed + rank)
        return env
    return _init

def train(args):
    # 1. Directories - Create run-specific subdirectories
    # Use provided run_name or generate one
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_drones{args.num_drones}_{timestamp}"
    
    # Create run-specific subdirectories
    # TensorBoard logs go to logs/tensorboard/{run_name} so all runs are visible together
    run_log_dir = os.path.join(args.log_dir, "tensorboard", run_name)
    run_model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    
    print(f"Run name: {run_name}")
    print(f"TensorBoard log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    # 2. Parallelization Config
    # Mac M-chips usually have 8+ cores. We use them all for data collection.
    num_cpu = multiprocessing.cpu_count()
    print(f"--- Mac Configuration Detected ---")
    print(f"Using {num_cpu} Parallel Environments (SubprocVecEnv)")
    
    # 3. Create Parallel Training Environments
    # This creates a list of distinct environment functions
    env_fns = [make_env(
        args.orders_csv, args.restaurants_json, args.pads_json, 
        args.num_drones, rank=i, seed=args.seed, is_eval=False,
        auto_chain_orders=args.auto_chain_orders,
        order_scale_factor=args.order_scale_factor
    ) for i in range(num_cpu)]

    # SubprocVecEnv runs each env in a separate process
    env = SubprocVecEnv(env_fns)
    
    # 4. Normalization
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 5. Evaluation Env (Keep simple/single-threaded for safety)
    # Use dynamic generation for eval too (same as training) for consistency
    eval_env = DummyVecEnv([make_env(
        args.orders_csv, args.restaurants_json, args.pads_json, 
        args.num_drones, rank=99, seed=args.seed+99, is_eval=False,  # Use dynamic generation
        auto_chain_orders=args.auto_chain_orders,
        order_scale_factor=args.order_scale_factor
    )])
    # Wrap with VecNormalize but sync stats from training env
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)
    # Sync normalization stats from training environment
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    # 6. Model Setup
    # n_steps is total steps per update. With parallel envs, divide by num_cpu for per-env steps.
    n_steps_per_env = args.n_steps // num_cpu
    if n_steps_per_env < 1:
        n_steps_per_env = 1
        print(f"Warning: n_steps ({args.n_steps}) is less than num_cpu ({num_cpu}). Using n_steps={num_cpu} per env.")
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=n_steps_per_env, # Per environment. Total = n_steps_per_env * num_cpu
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=run_log_dir,
        device="auto", # On Mac, this might pick 'mps' (Metal) or 'cpu'. Both are fine.
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    print(f"n_steps: {n_steps_per_env} per env × {num_cpu} envs = {n_steps_per_env * num_cpu} total steps per update")

    # 7. Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_model_dir, "best_model"),
        log_path=run_log_dir,
        eval_freq=max(100000 // num_cpu, 1), # Adjust freq based on parallelism
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500000 // num_cpu, 1),
        save_path=run_model_dir,
        name_prefix=run_name
    )

    print(f"Starting training for {args.total_timesteps} steps...")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save(os.path.join(run_model_dir, f"{run_name}_final"))
    env.save(os.path.join(run_model_dir, f"{run_name}_vec_normalize.pkl"))
    print("Training Complete.")
    print(f"All artifacts saved to: {run_model_dir} and {run_log_dir}")

if __name__ == "__main__":
    # MANDATORY FOR MAC OS MULTIPROCESSING
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-csv", type=str, default="orders.csv")
    parser.add_argument("--restaurants-json", type=str, default="config/restaurants.json")
    parser.add_argument("--pads-json", type=str, default="config/delivery_destinations.json")
    parser.add_argument("--num-drones", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=4096, 
                        help="Total steps per update (will be divided by number of parallel envs)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this training run (e.g., 'PPO_6'). All artifacts will be saved under this subdirectory.")
    parser.add_argument("--auto-chain-orders", action="store_true", default=False,
                        help="If set, drones automatically route restaurant->dropoff->base after pickup")
    parser.add_argument("--order-scale-factor", type=int, default=1,
                        help="Scale factor for orders (1 = all orders, 20 = 1/20 of orders)")
    
    args = parser.parse_args()
    train(args)