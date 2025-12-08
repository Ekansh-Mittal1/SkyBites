import numpy as np
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import STATUS_IDLE, STATUS_MOVING, STATUS_SERVICE, STATUS_CHARGING, STATUS_DEAD


class GreedyDispatcher:
    def __init__(
        self, 
        env,
        battery_threshold_percent: float = 20.0,
        battery_resume_percent: float = 80.0,
        max_delivery_time_minutes: float = 15.0
    ):
        self.env = env
        self.battery_threshold_percent = battery_threshold_percent
        self.battery_resume_percent = battery_resume_percent
        self.max_delivery_time_minutes = max_delivery_time_minutes
        self.base_node_id = env.base_node_id
        self.battery_capacity_wh = env.battery_capacity_wh
        self.drone_assignments = {}
    
    def reset(self):
        self.drone_assignments = {}
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        return geodesic(loc1, loc2).meters
    
    def _get_drone_location(self, drone_obs: np.ndarray) -> Tuple[float, float]:
        return (float(drone_obs[0]), float(drone_obs[1]))
    
    def _get_node_location(self, node_id: int) -> Optional[Tuple[float, float]]:
        return self.env._get_loc_from_id(node_id)
    
    def _get_order_urgency(self, order_obs: np.ndarray) -> float:
        age_minutes = float(order_obs[0])
        if age_minutes >= self.max_delivery_time_minutes:
            urgency = 1000.0 + (age_minutes - self.max_delivery_time_minutes) * 10.0
        else:
            urgency = age_minutes / self.max_delivery_time_minutes * 100.0
        return urgency
    
    def _should_charge(self, battery_wh: float) -> bool:
        battery_percent = (battery_wh / self.battery_capacity_wh) * 100.0
        return battery_percent < self.battery_threshold_percent
    
    def _is_at_base(self, drone_loc: Tuple[float, float]) -> bool:
        base_loc = self.env._get_base_loc()
        lat_diff = abs(drone_loc[0] - base_loc[0])
        lon_diff = abs(drone_loc[1] - base_loc[1])
        return lat_diff < 0.0001 and lon_diff < 0.0001
    
    def _find_best_order_for_drone(
        self, 
        drone_idx: int,
        drone_obs: np.ndarray,
        orders_obs: np.ndarray,
        battery_wh: float
    ) -> Optional[Tuple[int, float]]:
        drone_loc = self._get_drone_location(drone_obs)
        best_order_idx = None
        best_score = -np.inf
        
        for i, order_obs in enumerate(orders_obs):
            if order_obs[0] < 0:
                continue
            
            pickup_node_id = int(order_obs[1])
            dropoff_node_id = int(order_obs[2])
            
            if pickup_node_id < 0 or dropoff_node_id < 0:
                continue
            
            pickup_loc = self._get_node_location(pickup_node_id)
            dropoff_loc = self._get_node_location(dropoff_node_id)
            
            if pickup_loc is None or dropoff_loc is None:
                continue
            
            order_found = None
            for active_order in self.env.active_orders:
                restaurant_id = active_order.get('restaurant_id', '')
                if self.env._get_node_id_from_restaurant_id(restaurant_id) == pickup_node_id:
                    order_found = active_order
                    break
            
            if order_found is not None:
                can_complete, total_energy = self.env._can_complete_order(
                    self.env.drones[drone_idx], pickup_node_id, order_found
                )
                if not can_complete:
                    continue
            else:
                dist_to_pickup = self._calculate_distance(drone_loc, pickup_loc)
                dist_pickup_to_dropoff = self._calculate_distance(pickup_loc, dropoff_loc)
                base_loc = self.env._get_base_loc()
                dist_dropoff_to_base = self._calculate_distance(dropoff_loc, base_loc)
                
                try:
                    energy_1, _ = self.env.physics.calculate_energy_cost(
                        distance_meters=dist_to_pickup,
                        wind_speed=0.0,
                        payload_mass=0.0
                    )
                    energy_2, _ = self.env.physics.calculate_energy_cost(
                        distance_meters=dist_pickup_to_dropoff,
                        wind_speed=0.0,
                        payload_mass=1.0
                    )
                    energy_3, _ = self.env.physics.calculate_energy_cost(
                        distance_meters=dist_dropoff_to_base,
                        wind_speed=0.0,
                        payload_mass=0.0
                    )
                    total_energy = energy_1 + energy_2 + energy_3
                except:
                    total_energy = (dist_to_pickup + dist_pickup_to_dropoff + dist_dropoff_to_base) * 0.01
                
                if battery_wh < total_energy * 1.1:
                    continue
            
            urgency = self._get_order_urgency(order_obs)
            dist_to_pickup = self._calculate_distance(drone_loc, pickup_loc)
            distance_score = -dist_to_pickup / 1000.0
            score = urgency * 10.0 + distance_score
        
            if score > best_score:
                best_score = score
                best_order_idx = i
        
        if best_order_idx is not None:
            return (best_order_idx, best_score)
        return None
    
    def select_action(self, obs: Dict) -> np.ndarray:
        fleet_obs = obs['fleet']
        orders_obs = obs['orders']
        
        num_drones = len(fleet_obs)
        actions = np.zeros(num_drones, dtype=np.int32)
        assigned_orders = set()
        
        for drone_idx in range(num_drones):
            drone = fleet_obs[drone_idx]
            payload_kg = float(drone[4])
            if payload_kg < 0.1 and drone_idx in self.drone_assignments:
                del self.drone_assignments[drone_idx]
        
        for drone_idx in range(num_drones):
            drone = fleet_obs[drone_idx]
            status = int(drone[3])
            battery_wh = float(drone[2])
            payload_kg = float(drone[4])
            drone_loc = self._get_drone_location(drone)
            
            if status == STATUS_DEAD:
                actions[drone_idx] = self.base_node_id
                if drone_idx in self.drone_assignments:
                    del self.drone_assignments[drone_idx]
                continue
            
            if status != STATUS_IDLE:
                actions[drone_idx] = self.base_node_id
                continue
            
            if payload_kg > 0.1:
                if drone_idx in self.drone_assignments:
                    _, dropoff_node_id = self.drone_assignments[drone_idx]
                    actions[drone_idx] = dropoff_node_id
                else:
                    env_drone = self.env.drones[drone_idx]
                    if env_drone.get('current_order') is not None:
                        order = env_drone['current_order']
                        delivery_id = order.get('delivery_location_id')
                        if delivery_id:
                            dropoff_node_id = self.env._get_node_id_from_pad_id(delivery_id)
                            if dropoff_node_id is not None:
                                self.drone_assignments[drone_idx] = (None, dropoff_node_id)
                                actions[drone_idx] = dropoff_node_id
                                continue
                    actions[drone_idx] = self.base_node_id
                continue
            
            if self._should_charge(battery_wh):
                if self._is_at_base(drone_loc):
                    actions[drone_idx] = self.base_node_id
                else:
                    actions[drone_idx] = self.base_node_id
                    if drone_idx in self.drone_assignments:
                        del self.drone_assignments[drone_idx]
                continue
            
            best_order = self._find_best_order_for_drone(drone_idx, drone, orders_obs, battery_wh)
            
            if best_order is not None:
                order_idx, urgency_score = best_order
                
                if order_idx in assigned_orders:
                    available_orders = []
                    for i, order_obs in enumerate(orders_obs):
                        if i not in assigned_orders and order_obs[0] >= 0:
                            available_orders.append((i, order_obs))
                    
                    best_order = None
                    best_score = -np.inf
                    for i, order_obs in available_orders:
                        pickup_node_id = int(order_obs[1])
                        dropoff_node_id = int(order_obs[2])
                        
                        if pickup_node_id < 0 or dropoff_node_id < 0:
                            continue
                        
                        pickup_loc = self._get_node_location(pickup_node_id)
                        dropoff_loc = self._get_node_location(dropoff_node_id)
                        
                        if pickup_loc is None or dropoff_loc is None:
                            continue
                        
                        dist_to_pickup = self._calculate_distance(drone_loc, pickup_loc)
                        dist_pickup_to_dropoff = self._calculate_distance(pickup_loc, dropoff_loc)
                        base_loc = self.env._get_base_loc()
                        dist_dropoff_to_base = self._calculate_distance(dropoff_loc, base_loc)
                        
                        try:
                            energy_1, _ = self.env.physics.calculate_energy_cost(
                                distance_meters=dist_to_pickup,
                                wind_speed=0.0,
                                payload_mass=0.0
                            )
                            energy_2, _ = self.env.physics.calculate_energy_cost(
                                distance_meters=dist_pickup_to_dropoff,
                                wind_speed=0.0,
                                payload_mass=1.0
                            )
                            energy_3, _ = self.env.physics.calculate_energy_cost(
                                distance_meters=dist_dropoff_to_base,
                                wind_speed=0.0,
                                payload_mass=0.0
                            )
                            total_energy = energy_1 + energy_2 + energy_3
                        except:
                            total_energy = (dist_to_pickup + dist_pickup_to_dropoff + dist_dropoff_to_base) * 0.01
                        
                        if battery_wh < total_energy * 1.1:
                            continue
                        
                        urgency = self._get_order_urgency(order_obs)
                        distance_score = -dist_to_pickup / 1000.0
                        score = urgency * 10.0 + distance_score
                        
                        if score > best_score:
                            best_score = score
                            best_order = (i, score)
                
                if best_order is not None:
                    order_idx, _ = best_order
                    order_obs = orders_obs[order_idx]
                    pickup_node_id = int(order_obs[1])
                    dropoff_node_id = int(order_obs[2])
                    
                    actions[drone_idx] = pickup_node_id
                    assigned_orders.add(order_idx)
                    self.drone_assignments[drone_idx] = (pickup_node_id, dropoff_node_id)
                    continue
            
            actions[drone_idx] = self.base_node_id
        
        return actions


def run_greedy_baseline(
    env,
    orders_csv: str = None,
    max_steps: int = None,
    verbose: bool = True,
    render_every: int = 60
):
    dispatcher = GreedyDispatcher(env)
    
    obs, info = env.reset()
    dispatcher.reset()
    step = 0
    total_reward = 0.0
    
    if verbose:
        print("Starting greedy baseline dispatcher...")
    
    while True:
        actions = dispatcher.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        step += 1
        
        if verbose and render_every is not None and step % render_every == 0:
            env.render()
            print(f"Step {step}, Total Reward: {total_reward:.2f}")
        
        if terminated or truncated:
            break
        
        if max_steps is not None and step >= max_steps:
            break
    
    completed_order_info = getattr(env, 'completed_order_info', [])
    delays = [order_info['delay_minutes'] for order_info in completed_order_info] if completed_order_info else []
    
    metrics = {
        'total_steps': step,
        'total_reward': total_reward,
        'completed_orders': env.completed_orders,
        'active_orders': len(env.active_orders),
        'crashed_drones': sum(1 for d in env.drones if d['status'] == STATUS_DEAD)
    }
    
    if delays:
        metrics['avg_delay_minutes'] = sum(delays) / len(delays)
        metrics['median_delay_minutes'] = sorted(delays)[len(delays) // 2] if len(delays) > 0 else 0.0
        metrics['max_delay_minutes'] = max(delays)
        metrics['min_delay_minutes'] = min(delays)
        metrics['delay_std_minutes'] = (sum((d - metrics['avg_delay_minutes'])**2 for d in delays) / len(delays))**0.5
        metrics['late_orders'] = sum(1 for d in delays if d > 30.0)
        metrics['on_time_orders'] = sum(1 for d in delays if d <= 30.0)
        metrics['on_time_rate'] = metrics['on_time_orders'] / len(delays) * 100.0 if len(delays) > 0 else 0.0
    else:
        metrics['avg_delay_minutes'] = 0.0
        metrics['median_delay_minutes'] = 0.0
        metrics['max_delay_minutes'] = 0.0
        metrics['min_delay_minutes'] = 0.0
        metrics['delay_std_minutes'] = 0.0
        metrics['late_orders'] = 0
        metrics['on_time_orders'] = 0
        metrics['on_time_rate'] = 0.0
    
    if verbose:
        print("\n" + "="*50)
        print("Greedy Baseline Results")
        print("="*50)
        print(f"Total Steps: {metrics['total_steps']}")
        print(f"Total Reward: {metrics['total_reward']:.2f}")
        print(f"Completed Orders: {metrics['completed_orders']}")
        print(f"Remaining Active Orders: {metrics['active_orders']}")
        print(f"Crashed Drones: {metrics['crashed_drones']}")
        if delays:
            print(f"\nOrder Delay Statistics:")
            print(f"  Average Delay: {metrics['avg_delay_minutes']:.2f} minutes ({metrics['avg_delay_minutes']/60:.2f} hours)")
            print(f"  Median Delay: {metrics['median_delay_minutes']:.2f} minutes ({metrics['median_delay_minutes']/60:.2f} hours)")
            print(f"  Min Delay: {metrics['min_delay_minutes']:.2f} minutes")
            print(f"  Max Delay: {metrics['max_delay_minutes']:.2f} minutes ({metrics['max_delay_minutes']/60:.2f} hours)")
            print(f"  Std Deviation: {metrics['delay_std_minutes']:.2f} minutes")
            print(f"  On-Time Rate: {metrics['on_time_rate']:.1f}% ({metrics['on_time_orders']}/{len(delays)} orders ≤ 30 min)")
            print(f"  Late Orders: {metrics['late_orders']} (> 30 minutes)")
        print("="*50)
    
    return metrics


if __name__ == "__main__":
    import argparse
    from env import SkyBitesEnv
    
    parser = argparse.ArgumentParser(description='Run greedy baseline dispatcher')
    parser.add_argument('--orders', type=str, default=None, 
                       help='Path to orders CSV file (None for dynamic generation)')
    parser.add_argument('--restaurants', type=str, default='config/restaurants.json',
                       help='Path to restaurants JSON config')
    parser.add_argument('--pads', type=str, default='config/delivery_destinations.json',
                       help='Path to delivery pads JSON config')
    parser.add_argument('--base', type=str, default='config/base.json',
                       help='Path to base JSON config')
    parser.add_argument('--drones', type=int, default=10,
                       help='Number of drones in fleet')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum number of steps to run')
    parser.add_argument('--render-every', type=int, default=60,
                       help='Render environment every N steps')
    
    args = parser.parse_args()
    
    env = SkyBitesEnv(
        orders_csv=args.orders,
        restaurants_json=args.restaurants,
        pads_json=args.pads,
        base_json=args.base,
        num_drones=args.drones,
        dynamic_generation=(args.orders is None)
    )
    
    metrics = run_greedy_baseline(
        env,
        orders_csv=args.orders,
        max_steps=args.max_steps,
        verbose=True,
        render_every=args.render_every
    )
