import gymnasium as gym
from gymnasium import spaces
from gymnasium import Wrapper
import simpy
import numpy as np
import pandas as pd
import json
from geopy.distance import geodesic
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import contextily as ctx

# Import physics model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from drone_physics import DronePhysicsModel
from simulate_orders import generate_random_day

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

    def __init__(self, orders_csv: str = None, restaurants_json: str = None, pads_json: str = None, 
                 base_json: str = "config/base.json", num_drones: int = 10, dynamic_generation: bool = False,
                 auto_chain_orders: bool = False, order_scale_factor: int = 1):
        super().__init__()
        
        # Validate order_scale_factor
        if order_scale_factor < 1:
            raise ValueError("order_scale_factor must be >= 1")
        
        # Store new parameters
        self.auto_chain_orders = auto_chain_orders
        self.order_scale_factor = order_scale_factor
        
        # 1. Load Configuration ONCE to memory
        self.restaurants = self._load_restaurants(restaurants_json) if restaurants_json else {}
        self.pads = self._load_pads(pads_json) if pads_json else {}
        self.base = self._load_base(base_json)
        
        # Store configs as lists for the generator (needed for generate_random_day)
        # Load the raw JSON lists
        if restaurants_json:
            with open(restaurants_json, 'r') as f:
                self.restaurants_list = json.load(f)
        else:
            self.restaurants_list = []
        
        if pads_json:
            with open(pads_json, 'r') as f:
                self.pads_list = json.load(f)
        else:
            self.pads_list = []
        
        # Store dynamic generation flag
        self.dynamic_generation = dynamic_generation
        
        # Load initial data
        if not self.dynamic_generation and orders_csv:
            self.orders_df = pd.read_csv(orders_csv)
            # Strip whitespace from column names
            self.orders_df.columns = self.orders_df.columns.str.strip()
        else:
            self.orders_df = None  # Will be generated on reset
        
        # Create node ID mapping: 0..N-1 = restaurants, N..M-1 = pads, M = base
        self.restaurant_ids = sorted(self.restaurants.keys())
        self.pad_ids = sorted(self.pads.keys())
        self.num_restaurants = len(self.restaurant_ids)
        self.num_pads = len(self.pad_ids)
        self.num_nodes = self.num_restaurants + self.num_pads + 1  # +1 for base
        # Base is always the last node (num_nodes - 1)
        self.base_node_id = self.num_nodes - 1
        
        # 2. Initialize Physics
        # Use SkyRanger R70: supports 2.0 kg payload, 35 min endurance
        self.physics = DronePhysicsModel(drone_config="skyranger_r70")
        battery_specs = self.physics._get_battery_specs()
        # Calculate energy if not directly specified
        if battery_specs['energy_wh'] is None:
            # Calculate from capacity (mAh) and voltage (V)
            if battery_specs['capacity_mah'] and battery_specs['voltage_v']:
                self.battery_capacity_wh = (battery_specs['capacity_mah'] / 1000.0) * battery_specs['voltage_v']
            else:
                # Fallback to a reasonable default
                self.battery_capacity_wh = 200.0  # Wh
        else:
            self.battery_capacity_wh = float(battery_specs['energy_wh'])
        
        # 3. Fleet Configuration
        self.num_drones = num_drones
        
        # 4. Define Action Space (MultiDiscrete)
        # Each drone can choose: 0..num_nodes-1 (restaurants + pads + base)
        # Base is at node_id = num_nodes - 1
        self.action_space = spaces.MultiDiscrete([self.num_nodes] * self.num_drones)
        
        # 5. Define Observation Space
        # Fleet: [Lat, Lon, Battery_Wh, Status_Enum, Payload_kg] for each drone
        # Orders: [Order_Age_Min, Pickup_Loc_ID, Dropoff_Loc_ID] for up to MAX_ORDERS_IN_OBS orders
        battery_max = float(self.battery_capacity_wh)
        self.observation_space = spaces.Dict({
            "fleet": spaces.Box(
                low=np.array([[-90.0, -180.0, 0.0, float(STATUS_DEAD), 0.0]] * self.num_drones, dtype=np.float32),
                high=np.array([[90.0, 180.0, battery_max, float(STATUS_CHARGING), 5.0]] * self.num_drones, dtype=np.float32),
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
        self.assigned_orders = 0  # Orders assigned to drones
        self.failed_orders = 0  # Orders that couldn't be completed due to battery
        self.total_reward = 0.0
    
    def __getstate__(self):
        """
        Custom pickling to exclude SimPy generators which cannot be pickled.
        Used for multiprocessing compatibility on Mac.
        """
        state = self.__dict__.copy()
        # Remove SimPy environment and any generator references
        state['simpy_env'] = None
        # Clean drone states to remove any SimPy event references
        if 'drones' in state and state['drones']:
            cleaned_drones = []
            for drone in state['drones']:
                cleaned_drone = drone.copy()
                # Remove SimPy event references (they're generators)
                if 'action_event' in cleaned_drone:
                    cleaned_drone['action_event'] = None
                cleaned_drones.append(cleaned_drone)
            state['drones'] = cleaned_drones
        return state
    
    def __setstate__(self, state):
        """
        Custom unpickling. SimPy environment will be recreated in reset().
        """
        self.__dict__.update(state)
        # Ensure simpy_env is None (will be created in reset)
        self.simpy_env = None

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
        
        # 1. Handle Random Seeding
        # If dynamic, we rely on numpy's internal state or seed explicitly
        if seed is not None:
            np.random.seed(seed)
        
        # 2. Generate Orders (if dynamic generation is enabled)
        if self.dynamic_generation:
            # Generate a fresh day in-memory
            # This makes every episode unique!
            self.orders_df = generate_random_day(
                self.restaurants_list, 
                self.pads_list
            )
            # Strip whitespace from column names
            self.orders_df.columns = self.orders_df.columns.str.strip()
        
        # 3. Reset SimPy
        self.simpy_env = simpy.Environment()
        
        # 4. Reset Drones (all start at base)
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
                "flight_energy_req": None,  # Total energy for this flight
                "crash_penalized": False,  # Track if crash penalty has been applied
                "chaining": False,  # Track if drone is in automatic chaining mode
                "chain_target": None  # Next target in the chain (dropoff pad, then base)
            }
            self.drones.append(drone_state)
            
            # Start the persistent SimPy process for this drone
            self.simpy_env.process(self._drone_process(drone_state))
        
        # 3. Reset Orders
        self.active_orders = []  # Orders currently spawned but not picked up
        self.completed_orders = 0
        self.assigned_orders = 0
        self.failed_orders = 0
        self.total_reward = 0.0
        self._last_completed_count = 0  # Track for incremental reward
        self._last_assigned_count = 0  # Track for incremental reward (assignment)
        self._last_failed_count = 0  # Track for incremental reward
        self._step_failed_orders = 0  # Track failures in current step
        self._step_assigned_orders = 0  # Track assignments in current step
        
        # Start the order spawner process
        self.simpy_env.process(self._order_spawner())
        
        # Run SimPy for a tiny amount to initialize all processes (so action_events are created)
        self.simpy_env.run(until=0.001)
        
        return self._get_obs(), {}

    def _can_complete_order(self, drone: Dict, restaurant_node_id: int, order: Dict) -> Tuple[bool, float]:
        """
        Check if a drone has enough battery to complete an order.
        Calculates energy needed for: current location -> restaurant -> dropoff -> base
        
        Args:
            drone: Drone state dictionary
            restaurant_node_id: Node ID of the restaurant
            order: Order dictionary with delivery location info
        
        Returns:
            Tuple of (can_complete: bool, total_energy_needed: float)
        """
        # Get restaurant location
        restaurant_loc = self._get_loc_from_id(restaurant_node_id)
        if restaurant_loc is None:
            return False, float('inf')
        
        # Get delivery location
        delivery_id = order.get('delivery_location_id', '')
        delivery_node_id = self._get_node_id_from_pad_id(delivery_id)
        if delivery_node_id is None:
            return False, float('inf')
        
        delivery_loc = self._get_loc_from_id(delivery_node_id)
        if delivery_loc is None:
            return False, float('inf')
        
        # Get base location
        base_loc = (self.base['lat'], self.base['lon'])
        
        # Calculate distances
        current_loc = drone['loc']
        dist_to_restaurant = geodesic(current_loc, restaurant_loc).meters
        dist_restaurant_to_delivery = geodesic(restaurant_loc, delivery_loc).meters
        dist_delivery_to_base = geodesic(delivery_loc, base_loc).meters
        
        # Calculate energy for each leg
        # Leg 1: Current -> Restaurant (no payload)
        current_wind = 0.0  # No wind
        try:
            energy_1, _ = self.physics.calculate_energy_cost(
                distance_meters=dist_to_restaurant,
                wind_speed=current_wind,
                payload_mass=drone['payload']  # Current payload
            )
        except:
            energy_1 = dist_to_restaurant * 0.01  # Fallback
        
        # Leg 2: Restaurant -> Delivery (with 1 kg payload)
        try:
            energy_2, _ = self.physics.calculate_energy_cost(
                distance_meters=dist_restaurant_to_delivery,
                wind_speed=current_wind,
                payload_mass=1.0  # Order payload
            )
        except:
            energy_2 = dist_restaurant_to_delivery * 0.01  # Fallback
        
        # Leg 3: Delivery -> Base (no payload after dropoff)
        try:
            energy_3, _ = self.physics.calculate_energy_cost(
                distance_meters=dist_delivery_to_base,
                wind_speed=current_wind,
                payload_mass=0.0  # No payload after delivery
            )
        except:
            energy_3 = dist_delivery_to_base * 0.01  # Fallback
        
        total_energy = energy_1 + energy_2 + energy_3
        can_complete = drone['battery'] >= total_energy
        
        return can_complete, total_energy
    def step(self, actions):
        """
        Execute one step of the environment.
        
        Args:
            actions: Array of node IDs for each drone (or num_nodes to stay/charge)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # --- Phase 1: Assign Tasks with Safety Checks ---
        for drone_idx, target_node_id in enumerate(actions):
            drone = self.drones[drone_idx]
            
            # Skip drones that are in automatic chaining mode (they handle their own routing)
            if drone.get('chaining', False):
                continue
            
            # Only allow reassignment if drone is IDLE or CHARGING (can interrupt charging)
            if drone['status'] in [STATUS_IDLE, STATUS_CHARGING] and drone['action_event'] is not None:
                # Safety check: If target is a restaurant, verify the drone can complete the order
                target_node_type = self._get_node_type_from_id(target_node_id)
                
                if target_node_type == 'restaurant':
                    # --- FIX START: Look for UNASSIGNED orders only ---
                    # We strictly want an order that is NOT yet assigned.
                    order = self._find_order_at_restaurant(target_node_id, check_assigned=True)
                    
                    if order is not None:
                        # Check if drone can complete the full mission
                        can_complete, total_energy = self._can_complete_order(
                            drone, target_node_id, order
                        )
                        
                        if not can_complete:
                            # Drone cannot complete this order - mark as failed
                            order_still_active = any(
                                o.get('restaurant_id') == order.get('restaurant_id') and 
                                o.get('timestamp_minutes') == order.get('timestamp_minutes')
                                for o in self.active_orders
                            )
                            if order_still_active and not order.get('failed', False):
                                order['failed'] = True
                                self.failed_orders += 1
                                self._step_failed_orders += 1
                            continue
                        
                        # Order can be completed - assign it to this drone
                        # We know it is unassigned because we asked for check_assigned=True
                        order['assigned'] = True
                        self.assigned_orders += 1
                        self._step_assigned_orders += 1
                    # --- FIX END ---
                
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
        
        # Update tracking counters for next step
        self._last_completed_count = self.completed_orders
        self._last_assigned_count = self.assigned_orders
        self._step_assigned_orders = 0  # Reset step counter
        self._step_failed_orders = 0  # Reset step counter
        
        return obs, reward, terminated, False, {}

    def _drone_process(self, drone: Dict):
        """
        Persistent SimPy process for a single drone.
        Handles flight physics and task execution.
        """
        drone['action_event'] = self.simpy_env.event()
        
        while True:
            # 1. Wait for instruction from Agent (via step function) OR continue chaining
            if drone.get('chaining', False) and drone.get('chain_target') is not None:
                # Continue automatic chaining - don't wait for agent action
                target_id = drone['chain_target']
            else:
                # Normal mode: wait for agent action
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
                # Use physics model to calculate recharge (with fallback)
                remaining_wh = self.battery_capacity_wh - drone['battery']
                if remaining_wh > 0:
                    try:
                        recharge_sec, recharge_min, _ = self.physics.calculate_recharge_time(
                            current_charge_wh=drone['battery'],
                            target_charge_percent=100.0
                        )
                        # Charge proportionally for the step duration
                        charge_rate_wh_per_sec = remaining_wh / recharge_sec if recharge_sec > 0 else 0
                        charge_amount = min(remaining_wh, charge_rate_wh_per_sec * SIM_STEP_SECONDS)
                    except (ValueError, Exception):
                        # Fallback: use default 60W charger if recharge calculation fails
                        charge_rate_wh_per_sec = 60.0 / 3600.0  # 60W = 60 Wh/s = 0.0167 Wh/s
                        charge_amount = charge_rate_wh_per_sec * SIM_STEP_SECONDS
                    
                    drone['battery'] = min(
                        self.battery_capacity_wh,
                        drone['battery'] + charge_amount
                    )
                continue
            
            # Calculate distance
            dist_meters = geodesic(drone['loc'], target_loc).meters
            
            # 3. Use Physics Model
            current_wind = 0.0  # No wind
            
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
                # Only pick up orders that haven't been picked up yet
                order = self._find_order_at_restaurant(target_id, check_assigned=False)
                if order is not None and not order.get('picked_up', False):
                    # Mark as picked up IMMEDIATELY to prevent other drones from grabbing it
                    order['picked_up'] = True
                    drone['payload'] = 1.0  # Assume 1 kg per order
                    drone['current_order'] = order
                    drone['status'] = STATUS_SERVICE
                    # Remove order from active queue BEFORE yielding
                    if order in self.active_orders:
                        self.active_orders.remove(order)
                    # Service time: 30 seconds to pick up
                    yield self.simpy_env.timeout(30)
                    
                    # If auto-chaining is enabled, automatically proceed to dropoff
                    if self.auto_chain_orders:
                        delivery_id = order.get('delivery_location_id', '')
                        delivery_node_id = self._get_node_id_from_pad_id(delivery_id)
                        if delivery_node_id is not None:
                            drone['chaining'] = True
                            drone['chain_target'] = delivery_node_id
                            # Continue immediately to dropoff (don't wait for agent)
                            continue
            
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
                    
                    # If auto-chaining is enabled and still chaining, proceed to base
                    if self.auto_chain_orders and drone.get('chaining', False):
                        drone['chain_target'] = self.base_node_id
                        # Continue immediately to base (don't wait for agent)
                        continue
            
            elif target_node_type == 'base':
                # Arrived at base - can charge here
                # If no order and low battery, agent should keep drone here to charge
                # The charging logic is handled when agent sends drone to base again
                drone['status'] = STATUS_IDLE
                
                # If we were chaining, reset chaining state
                if drone.get('chaining', False):
                    drone['chaining'] = False
                    drone['chain_target'] = None
            
            # Reset action event for next command (only if not chaining)
            if not drone.get('chaining', False):
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

    def _find_order_at_restaurant(self, restaurant_node_id: int, check_assigned: bool = True) -> Optional[Dict]:
        """
        Find an active order waiting at a specific restaurant.
        FIXED: This creates a COPY or Reference but DOES NOT modify the 'assigned' flag.
        Assignment must happen in step(), not here.
        """
        if restaurant_node_id >= self.num_restaurants:
            return None
        
        restaurant_id = self.restaurant_ids[restaurant_node_id]
        
        # Find first order at this restaurant
        for order in self.active_orders:
            if order.get('restaurant_id') == restaurant_id:
                # Check if order is already assigned to a drone
                if check_assigned and order.get('assigned', False):
                    continue # Skip assigned orders
                
                # Found a valid order! Return it.
                # DO NOT SET order['assigned'] = True HERE!
                return order
                
        return None

    def _order_spawner(self):
        """SimPy process that spawns orders based on CSV timestamps."""
        # Safety check: ensure orders_df exists and is not empty
        if self.orders_df is None or len(self.orders_df) == 0:
            return
        
        # Sort dataframe by time
        sorted_orders = self.orders_df.sort_values('timestamp_minutes')
        
        # Systematic order filtering: only spawn every Nth order
        order_index = 0
        for _, row in sorted_orders.iterrows():
            # Only spawn if this order index matches the scale factor (every Nth order)
            if order_index % self.order_scale_factor != 0:
                order_index += 1
                continue
            order_index += 1
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
                'spawn_time': self.simpy_env.now,
                'lateness_penalty_accumulated': 0.0  # Track accumulated lateness penalty (capped at 20)
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
        +100 for each order assigned to a drone (incremental)
        -1 for each minute an order is late (age > 30 minutes, capped at -20 per order)
        -100 for each crashed drone
        -5 for each failed order (drone couldn't complete due to insufficient battery, incremental)
        """
        reward = 0.0
        
        # Reward for assigned orders (incremental - only count new assignments)
        # This gives reward when order is assigned, not when delivered
        reward += self._step_assigned_orders * 100.0
        
        # Penalty for failed orders (use step counter for immediate reward)
        reward -= self._step_failed_orders * 5.0
        
        # Penalty for old orders (capped at -20 per order)
        current_time_min = self.simpy_env.now / 60.0
        for order in self.active_orders:
            age_minutes = current_time_min - order.get('timestamp_minutes', 0)
            if age_minutes > 30:  # Late orders
                # Calculate penalty increment (per minute)
                penalty_increment = 0.1
                # Cap per order: ensure total doesn't exceed 20.0
                accumulated = order.get('lateness_penalty_accumulated', 0.0)
                remaining_capacity = 20.0 - accumulated
                penalty_increment = min(penalty_increment, max(0.0, remaining_capacity))
                
                # Update accumulated penalty and apply to reward
                if penalty_increment > 0:
                    order['lateness_penalty_accumulated'] = accumulated + penalty_increment
                    reward -= penalty_increment
        
        # Penalty for crashed drones (one-time penalty only)
        for drone in self.drones:
            if drone['status'] == STATUS_DEAD:
                # Check: Have we already fined this drone?
                if not drone['crash_penalized']:
                    reward -= 100.0  # Apply the big penalty ONCE
                    drone['crash_penalized'] = True  # Mark as paid
            else:
                # Optional: If you ever implement repair logic, reset the flag here
                pass
        
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
        Visualize the environment showing restaurants, pads, base, and drones with map background.
        
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
        
        # Set axis limits with padding
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        padding = max(lat_range, lon_range) * 0.1
        
        ax.set_xlim(min(all_lons) - padding, max(all_lons) + padding)
        ax.set_ylim(min(all_lats) - padding, max(all_lats) + padding)
        ax.set_aspect('equal', adjustable='box')
        
        # Add basemap using contextily
        # Contextily requires Web Mercator (EPSG:3857) coordinates
        # We'll convert the axis limits and add the basemap
        try:
            # Get current axis limits in lat/lon
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Convert to Web Mercator for contextily
            # Contextily expects EPSG:3857 (Web Mercator)
            # We'll use contextily's built-in conversion
            ctx.add_basemap(
                ax,
                crs='EPSG:4326',  # WGS84 (lat/lon)
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution_size=8
            )
        except Exception as e:
            # Fallback if contextily fails (e.g., no internet)
            print(f"Warning: Could not load basemap: {e}")
            print("Falling back to grid visualization")
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        return fig, ax


class DroneActionMaskWrapper(Wrapper):
    """
    SB3-Contrib compatible wrapper.
    Implements the required `action_masks()` method with smart physics-based masking.
    MOVED HERE FOR MULTIPROCESSING COMPATIBILITY ON MAC.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def action_masks(self):
        num_drones = self.env.num_drones
        num_nodes = self.env.num_nodes
        num_restaurants = self.env.num_restaurants
        base_node_id = num_nodes - 1
        
        masks = []
        unwrapped = self.env.unwrapped
        
        # Safety checks for initialization
        if not hasattr(unwrapped, 'drones') or not unwrapped.drones:
            return [np.ones(num_nodes, dtype=bool) for _ in range(num_drones)]
        
        try:
            for drone in unwrapped.drones:
                status = drone.get('status', STATUS_IDLE)
                is_chaining = drone.get('chaining', False)
                
                # Start with all actions BLOCKED
                drone_mask = np.zeros(num_nodes, dtype=bool)
                
                # ---------------------------------------------------------
                # CASE 1: Drone is locked (Chaining, Moving, Service, Dead)
                # ---------------------------------------------------------
                # Must force the "No-Op" action (Base/Stay)
                if is_chaining or status in [STATUS_MOVING, STATUS_SERVICE, STATUS_DEAD]:
                    drone_mask[base_node_id] = True
                    masks.append(drone_mask)
                    continue

                # ---------------------------------------------------------
                # CASE 2: Drone is Available (Idle or Charging)
                # ---------------------------------------------------------
                # 1. Always allow "Stay/Continue Charging" (Safety fallback)
                drone_mask[base_node_id] = True
                
                # 2. Check Feasibility for Deliveries
                # Even if charging, we only allow leaving if it can finish the job
                for node_id in range(num_restaurants):
                    # Check order existence and ownership
                    order = unwrapped._find_order_at_restaurant(node_id, check_assigned=True)
                    
                    if order:
                        # Physics Check: Loc -> Restaurant -> Dest -> Base
                        # This works for charging drones too (start_loc = base)
                        can_do, _ = unwrapped._can_complete_order(drone, node_id, order)
                        
                        if can_do:
                            drone_mask[node_id] = True
                
                masks.append(drone_mask)

        except Exception as e:
            # Emergency fallback to prevent crash, but default to STAY (safe)
            # rather than "Allow All" (unsafe)
            print(f"Mask Error: {e}")
            safe_mask = np.zeros(num_nodes, dtype=bool)
            safe_mask[base_node_id] = True
            masks = [safe_mask for _ in range(num_drones)]
            
        return masks
