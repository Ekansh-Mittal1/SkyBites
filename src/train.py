#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
import multiprocessing
from datetime import datetime
import numpy as np
import gymnasium as gym

warnings.filterwarnings('ignore', message='.*env.action_masks.*', category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SkyBitesEnv, DroneActionMaskWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

def make_env(orders_csv, restaurants_json, pads_json, num_drones, rank=0, seed=0, is_eval=False,
             auto_chain_orders=False, order_scale_factor=1):
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
        obs, info = env.reset(seed=seed + rank)
        return env
    return _init

def train_stage(args):
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_drones{args.num_drones}_{timestamp}"
    
    run_log_dir = os.path.join(args.log_dir, "tensorboard", run_name)
    run_model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    
    print(f"Run name: {run_name}")
    print(f"TensorBoard log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    num_cpu = multiprocessing.cpu_count()
    print(f"Using {num_cpu} Parallel Environments (SubprocVecEnv)")
    
    env_fns = [make_env(
        args.orders_csv, args.restaurants_json, args.pads_json, 
        args.num_drones, rank=i, seed=args.seed, is_eval=False,
        auto_chain_orders=args.auto_chain_orders,
        order_scale_factor=args.order_scale_factor
    ) for i in range(num_cpu)]

    env = SubprocVecEnv(env_fns)
    
    if args.load_stats and os.path.exists(args.load_stats):
        print(f"Loading VecNormalize stats from previous stage: {args.load_stats}")
        env = VecNormalize.load(args.load_stats, env)
        env.training = True
        print("Continuing to update normalization stats during training")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        if args.load_stats:
            print(f"Warning: Stats file not found: {args.load_stats}. Starting with fresh normalization.")

    eval_env = DummyVecEnv([make_env(
        args.orders_csv, args.restaurants_json, args.pads_json, 
        args.num_drones, rank=99, seed=args.seed+99, is_eval=False,
        auto_chain_orders=args.auto_chain_orders,
        order_scale_factor=args.order_scale_factor
    )])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)
    if hasattr(env, 'obs_rms') and env.obs_rms is not None:
        eval_env.obs_rms = env.obs_rms
    if hasattr(env, 'ret_rms') and env.ret_rms is not None:
        eval_env.ret_rms = env.ret_rms

    n_steps_per_env = args.n_steps // num_cpu
    if n_steps_per_env < 1:
        n_steps_per_env = 1
        print(f"Warning: n_steps ({args.n_steps}) is less than num_cpu ({num_cpu}). Using n_steps={num_cpu} per env.")
    
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model weights from previous stage: {args.load_model}")
        model = MaskablePPO.load(args.load_model, env=env)
        model.learning_rate = args.learning_rate
        if args.entropy_coef is not None:
            model.ent_coef = args.entropy_coef
        print(f"Updated learning_rate: {args.learning_rate}")
        if args.entropy_coef is not None:
            print(f"Updated entropy_coef: {args.entropy_coef}")
    else:
        model_kwargs = dict(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=n_steps_per_env,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
            tensorboard_log=run_log_dir,
            device="auto",
            policy_kwargs=dict(net_arch=[256, 256])
        )
        if args.entropy_coef is not None:
            model_kwargs['ent_coef'] = args.entropy_coef
        model = MaskablePPO(**model_kwargs)
        if args.load_model:
            print(f"Warning: Model file not found: {args.load_model}. Starting with fresh model.")
    
    print(f"n_steps: {n_steps_per_env} per env × {num_cpu} envs = {n_steps_per_env * num_cpu} total steps per update")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_model_dir, "best_model"),
        log_path=run_log_dir,
        eval_freq=max(100000 // num_cpu, 1),
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
    
    return run_name

def train_curriculum(num_drones=15, base_seed=42, log_dir="logs", model_dir="models",
                    restaurants_json="config/restaurants.json",
                    pads_json="config/delivery_destinations.json",
                    base_json="config/base.json",
                    run_prefix=""):
    stages = [
        {
            'name': f'{run_prefix}stage1_crawl',
            'order_scale_factor': 20,
            'entropy_coef': 0.01,
            'timesteps': 3_000_000,
            'load_model': None,
            'load_stats': None,
            'learning_rate': 3e-4,
            'batch_size': 256,
        },
        {
            'name': f'{run_prefix}stage2_walk',
            'order_scale_factor': 5,
            'entropy_coef': 0.005,
            'timesteps': 4_000_000,
            'load_model': None,
            'load_stats': None,
            'learning_rate': 2.5e-4,
            'batch_size': 256,
        },
        {
            'name': f'{run_prefix}stage3_run',
            'order_scale_factor': 1,
            'entropy_coef': 0.001,
            'timesteps': 5_000_000,
            'load_model': None,
            'load_stats': None,
            'learning_rate': 2e-4,
            'batch_size': 512,
        }
    ]
    
    print("="*80)
    print(f"CURRICULUM LEARNING - {num_drones} DRONES")
    print("="*80)
    print(f"Fleet Size: {num_drones} drones")
    print(f"Base Seed: {base_seed}")
    print(f"Total Stages: {len(stages)}")
    print("="*80)
    print()
    
    for stage_idx, stage in enumerate(stages, 1):
        print("\n" + "="*80)
        print(f"STAGE {stage_idx}/{len(stages)}: {stage['name'].upper()}")
        print("="*80)
        print(f"Order Scale Factor: {stage['order_scale_factor']} (~{1000//stage['order_scale_factor']} orders)")
        print(f"Entropy Coefficient: {stage['entropy_coef']}")
        print(f"Training Timesteps: {stage['timesteps']:,}")
        print(f"Learning Rate: {stage['learning_rate']}")
        print(f"Batch Size: {stage['batch_size']}")
        
        if stage_idx > 1:
            prev_stage = stages[stage_idx - 2]
            stage['load_model'] = os.path.join(model_dir, prev_stage['name'], f"{prev_stage['name']}_final.zip")
            stage['load_stats'] = os.path.join(model_dir, prev_stage['name'], f"{prev_stage['name']}_vec_normalize.pkl")
            print(f"Loading model from: {stage['load_model']}")
            print(f"Loading stats from: {stage['load_stats']}")
        
        class Args:
            pass
        
        args = Args()
        args.orders_csv = None
        args.restaurants_json = restaurants_json
        args.pads_json = pads_json
        args.base_json = base_json
        args.num_drones = num_drones
        args.total_timesteps = stage['timesteps']
        args.log_dir = log_dir
        args.model_dir = model_dir
        args.learning_rate = stage['learning_rate']
        args.n_steps = 4096
        args.batch_size = stage['batch_size']
        args.gamma = 0.999
        args.seed = base_seed
        args.run_name = stage['name']
        args.auto_chain_orders = True
        args.order_scale_factor = stage['order_scale_factor']
        args.entropy_coef = stage['entropy_coef']
        args.load_model = stage['load_model']
        args.load_stats = stage['load_stats']
        
        try:
            train_stage(args)
            model_filename = f"{stage['name']}_final.zip"
            stats_filename = f"{stage['name']}_vec_normalize.pkl"
            print(f"\n✓ Stage {stage_idx} completed successfully!")
            print(f"  Model saved: {os.path.join(model_dir, stage['name'], model_filename)}")
            print(f"  Stats saved: {os.path.join(model_dir, stage['name'], stats_filename)}")
        except Exception as e:
            print(f"\n✗ Stage {stage_idx} failed with error: {e}")
            raise
    
    print("\n" + "="*80)
    print(f"CURRICULUM LEARNING COMPLETE - {num_drones} DRONES!")
    print("="*80)
    print(f"All stages completed successfully.")
    final_model_name = f"{stages[-1]['name']}_final.zip"
    print(f"Final model: {os.path.join(model_dir, stages[-1]['name'], final_model_name)}")
    print("="*80)
    
    return os.path.join(model_dir, stages[-1]['name'], final_model_name)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Training script with curriculum learning')
    parser.add_argument("--num-drones", type=int, required=True, help="Number of drones in fleet")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--run-prefix", type=str, default="", help="Prefix for run names (e.g., 'train15_')")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for models")
    parser.add_argument("--restaurants-json", type=str, default="config/restaurants.json", help="Path to restaurants config")
    parser.add_argument("--pads-json", type=str, default="config/delivery_destinations.json", help="Path to delivery pads config")
    parser.add_argument("--base-json", type=str, default="config/base.json", help="Path to base config")
    
    args = parser.parse_args()
    
    train_curriculum(
        num_drones=args.num_drones,
        base_seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        restaurants_json=args.restaurants_json,
        pads_json=args.pads_json,
        base_json=args.base_json,
        run_prefix=args.run_prefix
    )

