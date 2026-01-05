import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SkyBitesEnv, DroneActionMaskWrapper

def make_eval_env(restaurants_json, pads_json, num_drones, 
                  auto_chain_orders, order_scale_factor):
    env = SkyBitesEnv(
        orders_csv=None,  # Not needed for dynamic generation
        restaurants_json=restaurants_json,
        pads_json=pads_json,
        num_drones=num_drones,
        dynamic_generation=True,  # Use dynamic generation like training
        auto_chain_orders=auto_chain_orders,
        order_scale_factor=order_scale_factor
    )
    env = DroneActionMaskWrapper(env)
    env = Monitor(env)  # Wrap with Monitor to track episode stats
    return env

def evaluate(args):
    print(f"--- Starting Evaluation ({args.episodes} Episodes) ---")
    print(f"Scale Factor: {args.order_scale_factor}")
    print(f"Using dynamic order generation (same as training)")
    
    # Create output directory
    outs_dir = "outs"
    os.makedirs(outs_dir, exist_ok=True)
    print(f"Output directory: {outs_dir}")
    
    # 1. Setup Environment with dynamic generation
    raw_env = make_eval_env(
        restaurants_json=args.restaurants_json,
        pads_json=args.pads_json,
        num_drones=args.num_drones,
        auto_chain_orders=args.auto_chain_orders,
        order_scale_factor=args.order_scale_factor
    )
    
    # 2. Load Stats
    env = DummyVecEnv([lambda: raw_env])
    try:
        env = VecNormalize.load(args.stats_path, env)
        env.training = False 
        env.norm_reward = False 
        print("Loaded VecNormalize stats successfully")
    except Exception as e:
        print(f"Warning: Normalization stats not loaded: {e}")

    # 3. Load Model
    model = MaskablePPO.load(args.model_path)
    results = []
    
    # Track all order events for logging
    all_order_events = []

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Get initial order count (before any steps)
        # The spawner spawns every Nth order starting at index 0, so count is ceiling division
        internal_env = env.envs[0].env.unwrapped
        if internal_env.orders_df is not None:
            df_len = len(internal_env.orders_df)
            scale = max(1, internal_env.order_scale_factor)
            # Ceiling division: (df_len + scale - 1) // scale
            episode_total_orders = (df_len + scale - 1) // scale
        else:
            episode_total_orders = 50  # Fallback estimate
        
        # Track order information
        order_info = {}  # order_id -> full order details
        order_assignments = {}  # order_id -> assignment info
        order_completions = {}  # order_id -> completion info
        order_failures = {}     # order_id -> failure info
        
        # Track previous state to detect changes
        prev_assigned_count = 0
        prev_completed_count = 0
        prev_failed_count = 0
        prev_active_orders = set()
        
        # Track completions incrementally (since env resets on done, we need to track ourselves)
        episode_completed_tracked = 0
        episode_failed_tracked = 0
        
        # Will capture completed_order_info incrementally
        episode_completed_order_info = []
        
        # Collect all orders that will spawn
        if internal_env.orders_df is not None:
            for idx, row in internal_env.orders_df.iterrows():
                if idx % internal_env.order_scale_factor == 0:
                    order_id = str(row.get('order_id', f'ORD_{idx}')).strip()
                    order_info[order_id] = {
                        'order_id': order_id,
                        'restaurant_id': str(row.get('restaurant_id', 'UNKNOWN')).strip(),
                        'delivery_location_id': str(row.get('delivery_location_id', 'UNKNOWN')).strip(),
                        'timestamp_minutes': float(row.get('timestamp_minutes', 0)),
                        'restaurant_lat': float(row.get('restaurant_latitude', 0)),
                        'restaurant_lon': float(row.get('restaurant_longitude', 0)),
                        'delivery_lat': float(row.get('delivery_latitude', 0)),
                        'delivery_lon': float(row.get('delivery_longitude', 0))
                    }
        
        while not done:
            # CRITICAL FIX: Get action masks and pass to predict
            # env.envs[0] is Monitor, env.envs[0].env is DroneActionMaskWrapper
            mask_wrapper = env.envs[0].env
            masks = mask_wrapper.action_masks()
            action, _ = model.predict(obs, deterministic=True, 
                                      action_masks=np.array(masks).flatten())
            
            # Track internal stats BEFORE step (this captures the state before auto-reset)
            internal_env = mask_wrapper.env.unwrapped
            current_time = internal_env.simpy_env.now / 60.0  # Convert to minutes
            
            # Track current active orders
            current_active_order_ids = set()
            for order in internal_env.active_orders:
                order_id = order.get('order_id', 'UNKNOWN')
                current_active_order_ids.add(order_id)
                
                # Update order info if we see it for the first time
                if order_id not in order_info:
                    order_info[order_id] = {
                        'order_id': order_id,
                        'restaurant_id': order.get('restaurant_id', 'UNKNOWN'),
                        'delivery_location_id': order.get('delivery_location_id', 'UNKNOWN'),
                        'timestamp_minutes': order.get('timestamp_minutes', 0),
                        'restaurant_lat': order.get('restaurant_lat', 0),
                        'restaurant_lon': order.get('restaurant_lon', 0),
                        'delivery_lat': order.get('delivery_lat', 0),
                        'delivery_lon': order.get('delivery_lon', 0)
                    }
            
            # Check for new assignments
            if internal_env.assigned_orders > prev_assigned_count:
                # Find which order was just assigned by checking active orders
                for order in internal_env.active_orders:
                    order_id = order.get('order_id', 'UNKNOWN')
                    if order.get('assigned', False) and order_id not in order_assignments:
                        # Find which drone was assigned
                        assigned_drone = None
                        restaurant_id = order.get('restaurant_id', 'UNKNOWN')
                        restaurant_node_id = internal_env._get_node_id_from_restaurant_id(restaurant_id)
                        
                        # Check which drone action targets this restaurant
                        if restaurant_node_id is not None:
                            action_array = action[0] if isinstance(action, (list, np.ndarray)) else action
                            for drone_idx, act in enumerate(action_array):
                                if act == restaurant_node_id and drone_idx < len(internal_env.drones):
                                    if internal_env.drones[drone_idx]['status'] in [0, 3]:  # IDLE or CHARGING
                                        assigned_drone = drone_idx
                                        break
                        
                        order_assignments[order_id] = {
                            'drone_id': assigned_drone,
                            'restaurant_id': restaurant_id,
                            'delivery_location_id': order.get('delivery_location_id', 'UNKNOWN'),
                            'timestamp_minutes': order.get('timestamp_minutes', 0),
                            'assigned_time': current_time,
                            'step': step_count
                        }
            
            # Check for new failures
            if internal_env.failed_orders > prev_failed_count:
                for order in internal_env.active_orders:
                    order_id = order.get('order_id', 'UNKNOWN')
                    if order.get('failed', False) and order_id not in order_failures:
                        order_failures[order_id] = {
                            'reason': 'Insufficient battery',
                            'timestamp': current_time,
                            'step': step_count,
                            'restaurant_id': order.get('restaurant_id', 'UNKNOWN'),
                            'delivery_location_id': order.get('delivery_location_id', 'UNKNOWN')
                        }
            
            # Check for completions (orders that were active but are no longer)
            completed_this_step = prev_active_orders - current_active_order_ids
            for order_id in completed_this_step:
                if order_id not in order_completions and order_id in order_assignments:
                    # Find which drone completed it
                    completed_drone = None
                    for drone_idx, drone in enumerate(internal_env.drones):
                        # Check if drone just completed (recently at a pad)
                        if drone['status'] == 2:  # STATUS_SERVICE (dropping off)
                            completed_drone = drone_idx
                    
                    order_completions[order_id] = {
                        'drone_id': completed_drone if completed_drone is not None else order_assignments[order_id].get('drone_id'),
                        'restaurant_id': order_assignments[order_id].get('restaurant_id', 'UNKNOWN'),
                        'delivery_location_id': order_assignments[order_id].get('delivery_location_id', 'UNKNOWN'),
                        'timestamp_minutes': order_assignments[order_id].get('timestamp_minutes', 0),
                        'completed_time': current_time,
                        'step': step_count
                    }
            
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            # Read current counts AFTER step (before environment potentially resets)
            current_completed = internal_env.completed_orders
            current_failed = internal_env.failed_orders
            
            # Track completions and failures incrementally
            if current_completed > prev_completed_count:
                episode_completed_tracked += (current_completed - prev_completed_count)
            if current_failed > prev_failed_count:
                episode_failed_tracked += (current_failed - prev_failed_count)
            
            # Update previous counts
            prev_assigned_count = internal_env.assigned_orders
            prev_completed_count = current_completed
            prev_failed_count = current_failed
            prev_active_orders = current_active_order_ids.copy()
            
            # Capture completed_order_info BEFORE environment potentially resets
            current_completed_order_info = getattr(internal_env, 'completed_order_info', [])
            if len(current_completed_order_info) > len(episode_completed_order_info):
                episode_completed_order_info = current_completed_order_info.copy()
            
            if done[0]:
                break
        
        # Use tracked values (captured incrementally during episode)
        episode_completed = episode_completed_tracked
        episode_failed = episode_failed_tracked
        
        # Extract delay statistics from captured completed_order_info
        delays = [order_info['delay_minutes'] for order_info in episode_completed_order_info] if episode_completed_order_info else []
        avg_delay = np.mean(delays) if delays else 0.0
        median_delay = np.median(delays) if delays else 0.0
        on_time_count = sum(1 for d in delays if d <= 0)  # On-time if delay <= 0
        on_time_rate = (on_time_count / len(delays) * 100) if delays else 0.0
        
        # Get final reward from Monitor wrapper (survives auto-reset)
        if 'episode' in info[0]:
            final_reward = info[0]['episode']['r']
        else:
            final_reward = total_reward
        
        total_spawned = episode_total_orders
        rate = (episode_completed / total_spawned * 100) if total_spawned > 0 else 0
        
        results.append({
            "Episode": episode + 1,
            "Total Reward": final_reward,
            "Orders Spawned": total_spawned,
            "Completed": episode_completed,
            "Failed": episode_failed,
            "Fulfillment Rate %": rate,
            "Avg Delay (min)": avg_delay,
            "Median Delay (min)": median_delay,
            "On-Time Rate %": on_time_rate
        })
        
        # Store episode data
        all_order_events.append({
            'episode': episode + 1,
            'order_info': order_info,
            'assignments': order_assignments,
            'completions': order_completions,
            'failures': order_failures
        })
        
        print(f"Episode {episode+1}: {episode_completed}/{total_spawned} Completed ({rate:.1f}%), Reward: {final_reward:.1f}")

    # Write detailed log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(outs_dir, f"eval_log_{args.run_name}_{timestamp}.txt")
    with open(log_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EVALUATION LOG: {args.run_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Order Scale Factor: {args.order_scale_factor}\n")
        f.write(f"Number of Drones: {args.num_drones}\n")
        f.write("="*80 + "\n\n")
        
        for ep_data in all_order_events:
            episode_num = ep_data['episode']
            f.write(f"\n{'='*80}\n")
            f.write(f"EPISODE {episode_num}\n")
            f.write(f"{'='*80}\n\n")
            
            # Write summary
            ep_result = results[episode_num - 1]
            f.write(f"Summary:\n")
            f.write(f"  Total Reward: {ep_result['Total Reward']:.1f}\n")
            f.write(f"  Orders Spawned: {ep_result['Orders Spawned']}\n")
            f.write(f"  Completed: {ep_result['Completed']}\n")
            f.write(f"  Failed: {ep_result['Failed']}\n")
            f.write(f"  Completion Rate: {ep_result['Fulfillment Rate %']:.1f}%\n\n")
            
            # Write all orders with their status
            f.write("Order Details:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Order ID':<15} {'Status':<12} {'Drone':<8} {'Restaurant':<12} {'Destination':<15} {'Order Time':<12} {'Event Time':<12}\n")
            f.write("-"*80 + "\n")
            
            order_info = ep_data['order_info']
            assignments = ep_data['assignments']
            completions = ep_data['completions']
            failures = ep_data['failures']
            
            for order_id, info in sorted(order_info.items()):
                status = "SPAWNED"
                drone_id = "N/A"
                event_time = 0.0
                
                if order_id in completions:
                    status = "COMPLETED"
                    drone_id = str(completions[order_id].get('drone_id', 'N/A'))
                    event_time = completions[order_id].get('completed_time', 0.0)
                elif order_id in failures:
                    status = "FAILED"
                    event_time = failures[order_id].get('timestamp', 0.0)
                elif order_id in assignments:
                    status = "ASSIGNED"
                    drone_id = str(assignments[order_id].get('drone_id', 'N/A'))
                    event_time = assignments[order_id].get('assigned_time', 0.0)
                
                f.write(f"{order_id:<15} "
                       f"{status:<12} "
                       f"{drone_id:<8} "
                       f"{info.get('restaurant_id', 'N/A'):<12} "
                       f"{info.get('delivery_location_id', 'N/A'):<15} "
                       f"{info.get('timestamp_minutes', 0):<12.1f} "
                       f"{event_time:<12.1f}\n")
            
            # Write detailed assignments
            if assignments:
                f.write(f"\nDetailed Assignments:\n")
                f.write("-"*80 + "\n")
                for order_id, assign_info in sorted(assignments.items()):
                    f.write(f"Order {order_id}:\n")
                    f.write(f"  Drone ID: {assign_info['drone_id']}\n")
                    f.write(f"  Restaurant ID: {assign_info['restaurant_id']}\n")
                    f.write(f"  Destination ID: {assign_info['delivery_location_id']}\n")
                    f.write(f"  Order Timestamp: {assign_info['timestamp_minutes']:.1f} minutes\n")
                    f.write(f"  Assigned At: {assign_info['assigned_time']:.1f} minutes (Step {assign_info['step']})\n")
                    if order_id in completions:
                        comp_info = completions[order_id]
                        f.write(f"  Completed At: {comp_info['completed_time']:.1f} minutes (Step {comp_info['step']})\n")
                        f.write(f"  Total Time: {comp_info['completed_time'] - assign_info['assigned_time']:.1f} minutes\n")
                    elif order_id in failures:
                        fail_info = failures[order_id]
                        f.write(f"  Failed At: {fail_info['timestamp']:.1f} minutes (Step {fail_info['step']})\n")
                        f.write(f"  Reason: {fail_info['reason']}\n")
                    f.write("\n")
    
    print(f"\nDetailed log saved to: {log_filename}")
    
    # Write summary report
    df = pd.DataFrame(results)
    summary_filename = os.path.join(outs_dir, f"eval_summary_{args.run_name}_{timestamp}.txt")
    with open(summary_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EVALUATION SUMMARY: {args.run_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write(f"Mean Reward: {df['Total Reward'].mean():.1f}\n")
        f.write(f"Mean Completion Rate: {df['Fulfillment Rate %'].mean():.1f}%\n")
        if 'Avg Delay (min)' in df.columns:
            f.write(f"Mean Avg Delay: {df['Avg Delay (min)'].mean():.1f} min\n")
            f.write(f"Mean Median Delay: {df['Median Delay (min)'].mean():.1f} min\n")
            f.write(f"Mean On-Time Rate: {df['On-Time Rate %'].mean():.1f}%\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {summary_filename}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("FINAL EVALUATION REPORT")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    print(f"Mean Reward: {df['Total Reward'].mean():.1f}")
    print(f"Mean Completion Rate: {df['Fulfillment Rate %'].mean():.1f}%")
    if 'Avg Delay (min)' in df.columns:
        print(f"Mean Avg Delay: {df['Avg Delay (min)'].mean():.1f} min")
        print(f"Mean Median Delay: {df['Median Delay (min)'].mean():.1f} min")
        print(f"Mean On-Time Rate: {df['On-Time Rate %'].mean():.1f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--restaurants-json", type=str, default="config/restaurants.json")
    parser.add_argument("--pads-json", type=str, default="config/delivery_destinations.json")
    parser.add_argument("--num-drones", type=int, default=10)
    parser.add_argument("--auto-chain-orders", action="store_true")
    parser.add_argument("--order-scale-factor", type=int, default=1)
    
    args = parser.parse_args()
    args.model_path = f"models/{args.run_name}/{args.run_name}_final.zip"
    args.stats_path = f"models/{args.run_name}/{args.run_name}_vec_normalize.pkl"
    
    evaluate(args)
