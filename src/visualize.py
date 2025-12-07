import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

# Try to import imageio for video creation
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not found. Install with: pip install imageio imageio-ffmpeg")
    print("Falling back to saving individual frames only.")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SkyBitesEnv, DroneActionMaskWrapper

def visualize_policy(model_path, stats_path, output_path=None, fps=10,
                     auto_chain_orders=True, order_scale_factor=20):
    print("Loading Visualization...")
    
    # Create output directory if not specified
    if output_path is None:
        # Extract run name from model path
        run_name = os.path.basename(os.path.dirname(model_path))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outs_dir = "outs"
        os.makedirs(outs_dir, exist_ok=True)
        output_path = os.path.join(outs_dir, f"visualization_{run_name}_{timestamp}.mp4")
    else:
        # If path is relative, ensure outs directory exists
        if not os.path.isabs(output_path) and not output_path.startswith("outs"):
            outs_dir = "outs"
            os.makedirs(outs_dir, exist_ok=True)
            output_path = os.path.join(outs_dir, output_path)
    
    print(f"Output video will be saved to: {output_path}")
    
    # 1. Setup Env with dynamic generation (same as training)
    raw_env = SkyBitesEnv(
        orders_csv=None,  # Not needed for dynamic generation
        restaurants_json="config/restaurants.json",
        pads_json="config/delivery_destinations.json",
        num_drones=10,
        dynamic_generation=True,  # Use dynamic generation like training
        auto_chain_orders=auto_chain_orders,
        order_scale_factor=order_scale_factor
    )
    # Apply Mask Wrapper
    raw_env = DroneActionMaskWrapper(raw_env)
    
    # Apply Normalization Wrapper (Crucial: Model sees normalized inputs)
    env = DummyVecEnv([lambda: raw_env])
    try:
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize stats successfully")
    except Exception as e:
        print(f"Warning: Could not load normalization stats ({e}). Visuals may be erratic.")

    # 2. Load Agent
    model = MaskablePPO.load(model_path)

    # 3. Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="drone_viz_")
    print(f"Saving frames to temporary directory: {temp_dir}")
    print(f"Output video will be saved to: {output_path}")
    
    # 4. Run Loop and collect frames
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    frame_paths = []
    
    print("Generating frames...")
    
    while not done:
        # CRITICAL FIX: Get action masks and pass to predict
        # env.envs[0] is the DroneActionMaskWrapper (has action_masks method)
        masks = env.envs[0].action_masks()
        action, _ = model.predict(obs, deterministic=True, 
                                  action_masks=np.array(masks).flatten())
        
        # Step Env
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step += 1
        
        # Access internal environment for visualization
        # env -> DummyVecEnv -> list of envs -> [0] -> DroneActionMaskWrapper -> SkyBitesEnv
        internal_env = env.envs[0].env.unwrapped
        
        # Save frame instead of displaying
        try:
            # Use non-interactive backend for saving
            fig, ax = internal_env.visualize(figsize=(10, 8), show_labels=True)
            
            # Save frame to temporary file
            frame_path = os.path.join(temp_dir, f"frame_{step:06d}.png")
            fig.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
            frame_paths.append(frame_path)
            plt.close(fig)  # Close to free memory
            
        except Exception as e:
            print(f"Warning: Error saving frame at step {step}: {e}")
            
        # Optional: Print status to console
        if step % 50 == 0:
            print(f"Step {step}: Time: {internal_env.simpy_env.now/60:.0f}m | "
                  f"Completed: {internal_env.completed_orders} | "
                  f"Active: {len(internal_env.active_orders)} | "
                  f"Reward: {total_reward:.1f}")

    print(f"\nEpisode Finished. Total Reward: {total_reward:.1f}")
    print(f"Completed Orders: {internal_env.completed_orders}")
    print(f"Total frames captured: {len(frame_paths)}")
    
    # 5. Compile frames into video
    if len(frame_paths) > 0:
        print(f"\nCompiling {len(frame_paths)} frames into video...")
        
        if HAS_IMAGEIO:
            try:
                # Read all frames
                frames = []
                for frame_path in frame_paths:
                    frames.append(imageio.imread(frame_path))
                
                # Write video
                imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
                print(f"Video saved successfully to: {output_path}")
                
            except Exception as e:
                print(f"Error creating video: {e}")
                print(f"Frames saved in: {temp_dir}")
                print("You can manually create a video using ffmpeg:")
                print(f"  ffmpeg -r {fps} -i {temp_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_path}")
        else:
            print(f"Frames saved in: {temp_dir}")
            print("Install imageio to automatically create video:")
            print("  pip install imageio imageio-ffmpeg")
            print(f"Or use ffmpeg manually:")
            print(f"  ffmpeg -r {fps} -i {temp_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_path}")
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files.")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
    else:
        print("No frames captured. Video not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained model and create video")
    parser.add_argument("run_name", type=str, help="Run name (e.g., 'PPO_7')")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video file path (default: visualization_<run_name>.mp4)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for output video (default: 10)")
    parser.add_argument("--model-dir", type=str, default="models", help="Base directory for models")
    parser.add_argument("--auto-chain-orders", action="store_true", default=True,
                        help="Enable automatic order chaining")
    parser.add_argument("--order-scale-factor", type=int, default=20,
                        help="Scale factor for orders (1 = all orders, 20 = 1/20 of orders)")
    
    args = parser.parse_args()
    
    # Construct paths based on run name
    model_path = os.path.join(args.model_dir, args.run_name, f"{args.run_name}_final.zip")
    stats_path = os.path.join(args.model_dir, args.run_name, f"{args.run_name}_vec_normalize.pkl")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(stats_path):
        print(f"Error: Stats file not found: {stats_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    print(f"Loading stats from: {stats_path}")
    
    visualize_policy(model_path, stats_path, output_path=args.output, fps=args.fps,
                     auto_chain_orders=args.auto_chain_orders,
                     order_scale_factor=args.order_scale_factor)
