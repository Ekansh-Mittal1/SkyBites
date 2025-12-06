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

def make_env(orders_csv, restaurants_json, pads_json, num_drones, rank=0, seed=0, is_eval=False):
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
    """
    def _init():
        env = SkyBitesEnv(
            orders_csv=orders_csv if is_eval else None,
            restaurants_json=restaurants_json,
            pads_json=pads_json,
            num_drones=num_drones,
            dynamic_generation=(not is_eval)
        )
        env = DroneActionMaskWrapper(env)
        env = Monitor(env, filename=None)
        # Reset must be called before any attribute access for multiprocessing
        obs, info = env.reset(seed=seed + rank)
        return env
    return _init

def train(args):
    # 1. Directories
    log_dir = args.log_dir
    model_dir = args.model_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_drones{args.num_drones}_{timestamp}"

    # 2. Parallelization Config
    # Mac M-chips usually have 8+ cores. We use them all for data collection.
    num_cpu = multiprocessing.cpu_count()
    print(f"--- Mac Configuration Detected ---")
    print(f"Using {num_cpu} Parallel Environments (SubprocVecEnv)")
    
    # 3. Create Parallel Training Environments
    # This creates a list of distinct environment functions
    env_fns = [make_env(
        args.orders_csv, args.restaurants_json, args.pads_json, 
        args.num_drones, rank=i, seed=args.seed, is_eval=False
    ) for i in range(num_cpu)]

    # SubprocVecEnv runs each env in a separate process
    env = SubprocVecEnv(env_fns)
    
    # 4. Normalization
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 5. Evaluation Env (Keep simple/single-threaded for safety)
    eval_env = DummyVecEnv([make_env(
        args.orders_csv, args.restaurants_json, args.pads_json, 
        args.num_drones, rank=99, seed=args.seed+99, is_eval=True
    )])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)

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
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="auto", # On Mac, this might pick 'mps' (Metal) or 'cpu'. Both are fine.
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    print(f"n_steps: {n_steps_per_env} per env × {num_cpu} envs = {n_steps_per_env * num_cpu} total steps per update")

    # 7. Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=log_dir,
        eval_freq=max(10000 // num_cpu, 1), # Adjust freq based on parallelism
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_cpu, 1),
        save_path=model_dir,
        name_prefix=run_name
    )

    print(f"Starting training for {args.total_timesteps} steps...")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save(os.path.join(model_dir, f"{run_name}_final"))
    env.save(os.path.join(model_dir, f"{run_name}_vec_normalize.pkl"))
    print("Training Complete.")

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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train(args)