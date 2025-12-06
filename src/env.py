import gymnasium as gym
from gymnasium import spaces
import simpy
import numpy as np
import pandas as pd
import json
from geopy.distance import geodesic
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Import physics model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from drone_physics import DronePhysicsModel

# Constants
SIM_STEP_SECONDS = 60  # Agent makes decisions every 1 minute
MAX_ORDERS_IN_OBS = 20  # Limit for observation space (pad if fewer, truncate if more)

# Status enums
STATUS_IDLE = 0
STATUS_MOVING = 1
STATUS_SERVICE = 2
STATUS_CHARGING = 3
STATUS_DEAD = -1


class SkyBitesEnv(gym.Env):
    """
    Gymnasium environment for drone delivery VRP simulation.
    Bridges discrete agent steps (every 60 seconds) with continuous SimPy events.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, orders_csv: str, restaurants_json: str, pads_json: str, base_json: str = "config/base.json", num_drones: int = 10):
        super().__init__()
        
        # 1. Load Configuration
        self.restaurants = self._load_restaurants(restaurants_json)
        self.pads = self._load_pads(pads_json)
        self.base = self._load_base(base_json)
        self.orders_df = pd.read_csv(orders_csv)
        # Strip whitespace from column names
        self.orders_df.columns = self.orders_df.columns.str.strip()
        
        # Create node ID mapping: 0..N-1 = restaurants, N..M-1 = pads, M = base
        self.restaurant_ids = sorted(self.restaurants.keys())
        self.pad_ids = sorted(self.pads.keys())
        self.num_restaurants = len(self.restaurant_ids)
        self.num_pads = len(self.pad_ids)
        self.num_nodes = self.num_restaurants + self.num_pads + 1  # +1 for base
        # Base is always the last node (num_nodes - 1)
        self.base_node_id = self.num_nodes - 1
        
        # 2. Initialize Physics
        self.physics = DronePhysicsModel(drone_name="dji-Mavic2")
        battery_specs = self.physics._get_battery_specs()
        self.battery_capacity_wh = battery_specs['energy_wh']
        
        # 3. Fleet Configuration
        self.num_drones = num_drones
        
        # 4. Define Action Space (MultiDiscrete)
        # Each drone can choose: 0..num_nodes-1 (restaurants + pads + base)
        # Base is at node_id = num_nodes - 1
        self.action_space = spaces.MultiDiscrete([self.num_nodes] * self.num_drones)
        
        # 5. Define Observation Space
        # Fleet: [Lat, Lon, Battery_Wh, Status_Enum, Payload_kg] for each drone
        # Orders: [Order_Age_Min, Pickup_Loc_ID, Dropoff_Loc_ID] for up to MAX_ORDERS_IN_OBS orders
        self.observation_space = spaces.Dict({
            "fleet": spaces.Box(
                low=np.array([[-90, -180, 0, STATUS_DEAD, 0]] * self.num_drones),
                high=np.array([[90, 180, self.battery_capacity_wh, STATUS_CHARGING, 5.0]] * self.num_drones),
                dtype=np.float32
            ),
            "orders": spaces.Box(
                low=-1,
                high=np.inf,
                shape=(MAX_ORDERS_IN_OBS, 3),
                dtype=np.float32
            )
        })
        
        # Initialize SimPy environment (will be reset in reset())
        self.simpy_env = None
        self.drones = []
        self.active_orders = []
        self.completed_orders = 0
        self.total_reward = 0.0

    def _load_restaurants(self, path: str) -> Dict[str, Dict]:
        """Load restaurants from JSON and create lookup dict."""
        with open(path, 'r') as f:
            restaurants_list = json.load(f)
        
        restaurants_dict = {}
        for r in restaurants_list:
            rid = r['restaurant_id']
            restaurants_dict[rid] = {
                'lat': r['location']['latitude'],
                'lon': r['location']['longitude'],
                'name': r.get('name', rid)
            }
        return restaurants_dict

    def _load_pads(self, path: str) -> Dict[str, Dict]:
        """Load delivery pads from JSON and create lookup dict."""
        with open(path, 'r') as f:
            pads_list = json.load(f)
        
        pads_dict = {}
        for p in pads_list:
            pid = p['destination_id']
            pads_dict[pid] = {
                'lat': p['location']['latitude'],
                'lon': p['location']['longitude'],
                'name': p.get('name', pid)
            }
        return pads_dict

    def _load_base(self, path: str) -> Dict:
        """Load base station from JSON."""
        with open(path, 'r') as f:
            base_data = json.load(f)
        
        return {
            'id': base_data.get('base_id', 'BASE_001'),
            'lat': base_data['location']['latitude'],
            'lon': base_data['location']['longitude'],
            'name': base_data.get('name', 'Drone Base Station')
        }

    def _get_loc_from_id(self, node_id: int) -> Optional[Tuple[float, float]]:
        """
        Convert node ID to (lat, lon) coordinates.
        node_id: 0..num_restaurants-1 = restaurants, 
                num_restaurants..num_nodes-2 = pads,
                num_nodes-1 = base
        Returns None if invalid ID
        """
        if node_id < 0 or node_id >= self.num_nodes:
            return None
        
        if node_id < self.num_restaurants:
            # Restaurant
            rid = self.restaurant_ids[node_id]
            r = self.restaurants[rid]
            return (r['lat'], r['lon'])
        elif node_id < self.num_nodes - 1:
            # Pad
            pad_idx = node_id - self.num_restaurants
            pid = self.pad_ids[pad_idx]
            p = self.pads[pid]
            return (p['lat'], p['lon'])
        else:
            # Base (last node)
            return (self.base['lat'], self.base['lon'])

    def _get_node_id_from_restaurant_id(self, restaurant_id: str) -> Optional[int]:
        """Get node ID for a restaurant ID string."""
        if restaurant_id in self.restaurant_ids:
            return self.restaurant_ids.index(restaurant_id)
        return None

    def _get_node_id_from_pad_id(self, pad_id: str) -> Optional[int]:
        """Get node ID for a pad ID string."""
        if pad_id in self.pad_ids:
            return self.num_restaurants + self.pad_ids.index(pad_id)
        return None

    def _get_base_loc(self) -> Tuple[float, float]:
        """Get base station location for initial drone placement."""
        return (self.base['lat'], self.base['lon'])

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # 1. Reset SimPy
        self.simpy_env = simpy.Environment()
        
        # 2. Reset Drones (all start at base)
        self.drones = []
        base_loc = self._get_base_loc()
        for i in range(self.num_drones):
            drone_state = {
                "id": i,
                "loc": base_loc,  # Start at base
                "battery": self.battery_capacity_wh,  # Start full
                "status": STATUS_IDLE,
                "payload": 0.0,
                "current_order": None,  # Order being delivered
                "action_event": None,  # Will be set in _drone_process
                # Flight interpolation fields
                "flight_start_time": None,
                "flight_duration": None,
                "start_loc": None,
                "target_loc": None,
                "flight_energy_req": None  # Total energy for this flight
            }
            self.drones.append(drone_state)
            
            # Start the persistent SimPy process for this drone
            self.simpy_env.process(self._drone_process(drone_state))
        
        # 3. Reset Orders
        self.active_orders = []  # Orders currently spawned but not picked up
        self.completed_orders = 0
        self.total_reward = 0.0
        
        # Start the order spawner process
        self.simpy_env.process(self._order_spawner())
        
        # Run SimPy for a tiny amount to initialize all processes (so action_events are created)
        self.simpy_env.run(until=0.001)
        
        return self._get_obs(), {}

    def step(self, actions):
        """
        Execute one step of the environment.
        
        Args:
            actions: Array of node IDs for each drone (or num_nodes to stay/charge)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # --- Phase 1: Assign Tasks ---
        for drone_idx, target_node_id in enumerate(actions):
            drone = self.drones[drone_idx]
            
            # Only allow reassignment if drone is IDLE
            if drone['status'] == STATUS_IDLE and drone['action_event'] is not None:
                # Wake up the drone process with new target
                if not drone['action_event'].triggered:
                    drone['action_event'].succeed(value=target_node_id)
                # Create new event for next action
                drone['action_event'] = self.simpy_env.event()
        
        # --- Phase 2: Run Physics/Time ---
        # Run SimPy for exactly SIM_STEP_SECONDS
        current_time = self.simpy_env.now
        self.simpy_env.run(until=current_time + SIM_STEP_SECONDS)
        
        # --- Phase 3: Observe & Reward ---
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self.simpy_env.now >= 24 * 3600  # End of day (24 hours)
        
        return obs, reward, terminated, False, {}

    def _drone_process(self, drone: Dict):
        """
        Persistent SimPy process for a single drone.
        Handles flight physics and task execution.
        """
        drone['action_event'] = self.simpy_env.event()
        
        while True:
            # 1. Wait for instruction from Agent (via step function)
            drone['status'] = STATUS_IDLE
            target_id = yield drone['action_event']
            
            # 2. Identify Target Coordinates
            target_loc = self._get_loc_from_id(target_id)
            if target_loc is None:
                continue  # Invalid target, skip
            
            # Check if target is base and we're already there
            target_node_type = self._get_node_type_from_id(target_id)
            current_node_type = self._get_node_type_from_loc(drone['loc'])
            
            if target_node_type == 'base' and current_node_type == 'base':
                # Already at base, charge
                drone['status'] = STATUS_CHARGING
                # Charge for the step duration
                yield self.simpy_env.timeout(SIM_STEP_SECONDS)
                # Use physics model to calculate recharge
                remaining_wh = self.battery_capacity_wh - drone['battery']
                if remaining_wh > 0:
                    recharge_sec, recharge_min, _ = self.physics.calculate_recharge_time(
                        current_charge_wh=drone['battery'],
                        target_charge_percent=100.0
                    )
                    # Charge proportionally for the step duration
                    charge_rate_wh_per_sec = remaining_wh / recharge_sec if recharge_sec > 0 else 0
                    charge_amount = min(remaining_wh, charge_rate_wh_per_sec * SIM_STEP_SECONDS)
                    drone['battery'] = min(
                        self.battery_capacity_wh,
                        drone['battery'] + charge_amount
                    )
                continue
            
            # Calculate distance
            dist_meters = geodesic(drone['loc'], target_loc).meters
            
            # 3. Use Physics Model
            current_wind = np.random.normal(5.0, 2.0)  # Random wind ~5 m/s
            current_wind = max(0.0, current_wind)  # No negative wind
            
            try:
                energy_req, flight_time = self.physics.calculate_energy_cost(
                    distance_meters=dist_meters,
                    wind_speed=current_wind,
                    payload_mass=drone['payload']
                )
            except Exception as e:
                # Fallback if physics calculation fails
                print(f"Warning: Physics calculation failed: {e}")
                energy_req = dist_meters * 0.01  # Rough estimate: 0.01 Wh per meter
                flight_time = dist_meters / 10.0  # Assume 10 m/s speed
            
            # 4. Check Feasibility
            if drone['battery'] < energy_req:
                # FAIL STATE: Drone crashes or forced landing
                drone['battery'] = 0
                drone['status'] = STATUS_DEAD
                yield self.simpy_env.timeout(999999)  # Stuck forever
                continue
            
            # 5. Execute Flight (Time Delay)
            drone['status'] = STATUS_MOVING
            
            # Store flight details for interpolation
            drone['flight_start_time'] = self.simpy_env.now
            drone['flight_duration'] = flight_time
            drone['start_loc'] = drone['loc']  # Current location (start of flight)
            drone['target_loc'] = target_loc  # Destination
            drone['flight_energy_req'] = energy_req  # Total energy for this flight
            
            # Wait for flight to complete
            yield self.simpy_env.timeout(flight_time)
            
            # 6. Arrival Updates
            drone['battery'] -= energy_req
            drone['loc'] = target_loc
            
            # Clear flight interpolation fields
            drone['flight_start_time'] = None
            drone['flight_duration'] = None
            drone['start_loc'] = None
            drone['target_loc'] = None
            drone['flight_energy_req'] = None
            
            # 7. Handle Pickup/Dropoff/Charging Logic
            target_node_type = self._get_node_type_from_id(target_id)
            
            if target_node_type == 'restaurant':
                # Try to pick up an order from this restaurant
                order = self._find_order_at_restaurant(target_id)
                if order is not None:
                    drone['payload'] = 1.0  # Assume 1 kg per order
                    drone['current_order'] = order
                    drone['status'] = STATUS_SERVICE
                    # Service time: 30 seconds to pick up
                    yield self.simpy_env.timeout(30)
                    # Remove order from active queue
                    if order in self.active_orders:
                        self.active_orders.remove(order)
            
            elif target_node_type == 'pad':
                # Drop off order if carrying one
                if drone['current_order'] is not None:
                    # Delivery complete!
                    self.completed_orders += 1
                    drone['current_order'] = None
                    drone['payload'] = 0.0
                    drone['status'] = STATUS_SERVICE
                    # Service time: 30 seconds to drop off
                    yield self.simpy_env.timeout(30)
            
            elif target_node_type == 'base':
                # Arrived at base - can charge here
                # If no order and low battery, agent should keep drone here to charge
                # The charging logic is handled when agent sends drone to base again
                drone['status'] = STATUS_IDLE
            
            # Reset action event for next command
            drone['action_event'] = self.simpy_env.event()

    def _get_node_type_from_id(self, node_id: int) -> Optional[str]:
        """Get node type ('restaurant', 'pad', or 'base') from node ID."""
        if node_id < 0 or node_id >= self.num_nodes:
            return None
        if node_id < self.num_restaurants:
            return 'restaurant'
        elif node_id < self.num_nodes - 1:
            return 'pad'
        else:
            return 'base'

    def _get_node_type_from_loc(self, loc: Tuple[float, float]) -> Optional[str]:
        """Determine if location is a restaurant, pad, or base."""
        lat, lon = loc
        # Check base first (most specific)
        if abs(self.base['lat'] - lat) < 0.0001 and abs(self.base['lon'] - lon) < 0.0001:
            return 'base'
        # Check restaurants
        for r in self.restaurants.values():
            if abs(r['lat'] - lat) < 0.0001 and abs(r['lon'] - lon) < 0.0001:
                return 'restaurant'
        # Check pads
        for p in self.pads.values():
            if abs(p['lat'] - lat) < 0.0001 and abs(p['lon'] - lon) < 0.0001:
                return 'pad'
        return None

    def _find_order_at_restaurant(self, restaurant_node_id: int) -> Optional[Dict]:
        """Find an active order waiting at a specific restaurant."""
        if restaurant_node_id >= self.num_restaurants:
            return None
        
        restaurant_id = self.restaurant_ids[restaurant_node_id]
        
        # Find first order at this restaurant that hasn't been assigned
        for order in self.active_orders:
            if order.get('restaurant_id') == restaurant_id:
                # Check if order is already assigned to a drone
                if not order.get('assigned', False):
                    order['assigned'] = True
                    return order
        return None

    def _order_spawner(self):
        """SimPy process that spawns orders based on CSV timestamps."""
        # Sort dataframe by time
        sorted_orders = self.orders_df.sort_values('timestamp_minutes')
        
        for _, row in sorted_orders.iterrows():
            spawn_time = float(row['timestamp_minutes']) * 60  # Convert to seconds
            
            # Wait until it's time for this order
            wait_time = spawn_time - self.simpy_env.now
            if wait_time > 0:
                yield self.simpy_env.timeout(wait_time)
            
            # Add to active queue
            order_dict = {
                'order_id': str(row.get('order_id', '')).strip(),
                'restaurant_id': str(row.get('restaurant_id', '')).strip(),
                'restaurant_lat': float(row.get('restaurant_latitude', 0)),
                'restaurant_lon': float(row.get('restaurant_longitude', 0)),
                'delivery_lat': float(row.get('delivery_latitude', 0)),
                'delivery_lon': float(row.get('delivery_longitude', 0)),
                'delivery_location_id': str(row.get('delivery_location_id', '')).strip(),
                'timestamp_minutes': float(row.get('timestamp_minutes', 0)),
                'assigned': False,
                'spawn_time': self.simpy_env.now
            }
            self.active_orders.append(order_dict)

    def _get_obs(self) -> Dict:
        """Get current observation for the agent."""
        # Fleet observation: [Lat, Lon, Battery_Wh, Status_Enum, Payload_kg]
        fleet_obs = np.zeros((self.num_drones, 5), dtype=np.float32)
        for i, drone in enumerate(self.drones):
            # Interpolate position if drone is moving
            if drone['status'] == STATUS_MOVING and drone['flight_start_time'] is not None:
                # Calculate progress through flight
                elapsed = self.simpy_env.now - drone['flight_start_time']
                progress = min(1.0, elapsed / drone['flight_duration']) if drone['flight_duration'] > 0 else 1.0
                
                # Linear interpolation of position
                start_lat, start_lon = drone['start_loc']
                end_lat, end_lon = drone['target_loc']
                
                current_lat = start_lat + (end_lat - start_lat) * progress
                current_lon = start_lon + (end_lon - start_lon) * progress
                
                fleet_obs[i, 0] = current_lat
                fleet_obs[i, 1] = current_lon
                
                # Interpolate battery drain so agent sees it dropping
                if drone['flight_energy_req'] is not None:
                    fleet_obs[i, 2] = drone['battery'] - (drone['flight_energy_req'] * progress)
                else:
                    fleet_obs[i, 2] = drone['battery']
            else:
                # Not moving, use actual location
                fleet_obs[i, 0] = drone['loc'][0]  # Lat
                fleet_obs[i, 1] = drone['loc'][1]  # Lon
                fleet_obs[i, 2] = drone['battery']  # Battery
            
            fleet_obs[i, 3] = drone['status']  # Status
            fleet_obs[i, 4] = drone['payload']  # Payload
        
        # Orders observation: [Order_Age_Min, Pickup_Loc_ID, Dropoff_Loc_ID]
        orders_obs = np.full((MAX_ORDERS_IN_OBS, 3), -1.0, dtype=np.float32)
        
        # Get up to MAX_ORDERS_IN_OBS active orders
        num_orders = min(len(self.active_orders), MAX_ORDERS_IN_OBS)
        current_time_min = self.simpy_env.now / 60.0
        
        for i in range(num_orders):
            order = self.active_orders[i]
            orders_obs[i, 0] = current_time_min - order.get('timestamp_minutes', 0)  # Age in minutes
            
            # Pickup location ID (restaurant node ID)
            restaurant_id = order.get('restaurant_id', '')
            pickup_node_id = self._get_node_id_from_restaurant_id(restaurant_id)
            if pickup_node_id is not None:
                orders_obs[i, 1] = float(pickup_node_id)
            
            # Dropoff location ID (pad node ID)
            delivery_id = order.get('delivery_location_id', '')
            dropoff_node_id = self._get_node_id_from_pad_id(delivery_id)
            if dropoff_node_id is not None:
                orders_obs[i, 2] = float(dropoff_node_id)
        
        return {
            "fleet": fleet_obs,
            "orders": orders_obs
        }

    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current step.
        +10 for each completed delivery
        -1 for each minute an order is late (age > 30 minutes)
        -100 for each crashed drone
        """
        reward = 0.0
        
        # Reward for completed orders (handled in _drone_process, but we track here)
        # Note: completed_orders is incremented in _drone_process
        
        # Penalty for old orders
        current_time_min = self.simpy_env.now / 60.0
        for order in self.active_orders:
            age_minutes = current_time_min - order.get('timestamp_minutes', 0)
            if age_minutes > 30:  # Late orders
                reward -= 1.0
        
        # Penalty for crashed drones
        for drone in self.drones:
            if drone['status'] == STATUS_DEAD:
                reward -= 100.0
        
        return reward

    def render(self, mode="human"):
        """Render the environment (optional)."""
        if mode == "human":
            print(f"Time: {self.simpy_env.now / 60.0:.1f} minutes")
            print(f"Active Orders: {len(self.active_orders)}")
            print(f"Completed Orders: {self.completed_orders}")
            for i, drone in enumerate(self.drones):
                status_str = ["IDLE", "MOVING", "SERVICE", "CHARGING", "DEAD"][drone['status'] + 1]
                print(f"Drone {i}: {status_str}, Battery: {drone['battery']:.1f} Wh, "
                      f"Loc: ({drone['loc'][0]:.4f}, {drone['loc'][1]:.4f})")

    def visualize(self, figsize=(12, 10), show_labels=True, show_drone_paths=False):
        """
        Visualize the environment showing restaurants, pads, base, and drones.
        
        Args:
            figsize: Figure size tuple (width, height)
            show_labels: Whether to show location labels
            show_drone_paths: Whether to show drone flight paths (if available)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Collect all locations for axis limits
        all_lats = []
        all_lons = []
        
        # Plot restaurants
        restaurant_lats = []
        restaurant_lons = []
        restaurant_names = []
        for rid, r in self.restaurants.items():
            restaurant_lats.append(r['lat'])
            restaurant_lons.append(r['lon'])
            restaurant_names.append(r.get('name', rid))
            all_lats.append(r['lat'])
            all_lons.append(r['lon'])
        
        ax.scatter(restaurant_lons, restaurant_lats, c='red', marker='s', s=150, 
                   label='Restaurants', zorder=4, edgecolors='darkred', linewidths=1.5)
        
        # Plot pads (delivery destinations)
        pad_lats = []
        pad_lons = []
        pad_names = []
        for pid, p in self.pads.items():
            pad_lats.append(p['lat'])
            pad_lons.append(p['lon'])
            pad_names.append(p.get('name', pid))
            all_lats.append(p['lat'])
            all_lons.append(p['lon'])
        
        ax.scatter(pad_lons, pad_lats, c='blue', marker='^', s=120, 
                   label='Delivery Pads', zorder=4, edgecolors='darkblue', linewidths=1.5)
        
        # Plot base
        base_lat = self.base['lat']
        base_lon = self.base['lon']
        all_lats.append(base_lat)
        all_lons.append(base_lon)
        
        ax.scatter(base_lon, base_lat, c='green', marker='*', s=300, 
                   label='Base Station', zorder=5, edgecolors='darkgreen', linewidths=2)
        
        # Plot drones with status-based colors
        if hasattr(self, 'drones') and len(self.drones) > 0:
            drone_lats = []
            drone_lons = []
            drone_statuses = []
            drone_ids = []
            
            for drone in self.drones:
                # Use interpolated position if moving
                if drone['status'] == STATUS_MOVING and drone.get('flight_start_time') is not None:
                    # Calculate interpolated position
                    elapsed = self.simpy_env.now - drone['flight_start_time']
                    progress = min(1.0, elapsed / drone['flight_duration']) if drone['flight_duration'] > 0 else 1.0
                    start_lat, start_lon = drone['start_loc']
                    end_lat, end_lon = drone['target_loc']
                    current_lat = start_lat + (end_lat - start_lat) * progress
                    current_lon = start_lon + (end_lon - start_lon) * progress
                    drone_lats.append(current_lat)
                    drone_lons.append(current_lon)
                else:
                    drone_lats.append(drone['loc'][0])
                    drone_lons.append(drone['loc'][1])
                
                drone_statuses.append(drone['status'])
                drone_ids.append(drone['id'])
                all_lats.append(drone['loc'][0])
                all_lons.append(drone['loc'][1])
            
            # Plot drones by status
            status_colors = {
                STATUS_IDLE: 'gray',
                STATUS_MOVING: 'orange',
                STATUS_SERVICE: 'purple',
                STATUS_CHARGING: 'cyan',
                STATUS_DEAD: 'black'
            }
            status_labels = {
                STATUS_IDLE: 'Idle',
                STATUS_MOVING: 'Moving',
                STATUS_SERVICE: 'Service',
                STATUS_CHARGING: 'Charging',
                STATUS_DEAD: 'Dead'
            }
            
            # Group drones by status for legend
            plotted_statuses = set()
            for i, (lat, lon, status, drone_id) in enumerate(zip(drone_lats, drone_lons, drone_statuses, drone_ids)):
                color = status_colors.get(status, 'gray')
                label = status_labels.get(status, 'Unknown')
                
                if status not in plotted_statuses:
                    ax.scatter(lon, lat, c=color, marker='o', s=200, 
                              label=f'Drones ({label})', zorder=6, edgecolors='black', linewidths=1.5)
                    plotted_statuses.add(status)
                else:
                    ax.scatter(lon, lat, c=color, marker='o', s=200, 
                              zorder=6, edgecolors='black', linewidths=1.5)
                
                # Add drone ID label
                ax.annotate(f'D{drone_id}', (lon, lat), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Add labels for restaurants and pads if requested
        if show_labels:
            # Label restaurants
            for lon, lat, name in zip(restaurant_lons, restaurant_lats, restaurant_names):
                ax.annotate(name, (lon, lat), xytext=(5, -15), 
                           textcoords='offset points', fontsize=7, color='darkred', alpha=0.7)
            
            # Label pads (optional, can be cluttered)
            # for lon, lat, name in zip(pad_lons, pad_lats, pad_names):
            #     ax.annotate(name, (lon, lat), xytext=(5, -15), 
            #                textcoords='offset points', fontsize=6, color='darkblue', alpha=0.6)
            
            # Label base
            ax.annotate('BASE', (base_lon, base_lat), xytext=(5, 15), 
                       textcoords='offset points', fontsize=10, fontweight='bold', color='darkgreen')
        
        # Set axis labels and title
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add time and stats to title
        if hasattr(self, 'simpy_env') and self.simpy_env is not None:
            time_min = self.simpy_env.now / 60.0
            completed = getattr(self, 'completed_orders', 0)
            active = len(getattr(self, 'active_orders', []))
            ax.set_title(f'Drone Delivery Environment\nTime: {time_min:.1f} min | '
                        f'Completed: {completed} | Active Orders: {active}', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Drone Delivery Environment', fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio and add padding
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        padding = max(lat_range, lon_range) * 0.1
        
        ax.set_xlim(min(all_lons) - padding, max(all_lons) + padding)
        ax.set_ylim(min(all_lats) - padding, max(all_lats) + padding)
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        return fig, ax
