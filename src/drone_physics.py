import drone_awe
import numpy as np

class DronePhysicsModel:
    def __init__(self, drone_name="dji-Mavic2", cruise_altitude=100.0, climb_rate=3.0, descent_rate=2.0):
        """
        Initialize the drone physics model.
        
        Args:
            drone_name: Valid drone name from drone-awe library.
                       Options: '3DR-IRIS', '3DR-Solo', 'aeryon-skyrangerR70',
                                'asctec-Falcon8', 'dji-Mavic2', 'dji-Phantom4RTK',
                                'freefly-alta8', 'aerovironment-Puma3AE',
                                'precisionhawk-FireFLY6Pro', 'senseFly-ebeeX',
                                'yuneec-TyphoonHPlus'
            cruise_altitude: Cruise altitude in meters (default: 100m)
            climb_rate: Vertical climb speed in m/s (default: 3.0 m/s)
            descent_rate: Vertical descent speed in m/s (default: 2.0 m/s, slower for safety)
        """
        self.drone_name = drone_name
        self.cruise_altitude = cruise_altitude
        self.climb_rate = climb_rate
        self.descent_rate = descent_rate
        # Default mission speed in m/s (can be overridden)
        self.default_speed = 10.0  # m/s
        # Load battery specs immediately before any models can corrupt the database
        self._battery_specs = self._get_battery_specs_early()

    def _get_battery_specs_early(self):
        """
        Load battery specs early, before any models run (which can corrupt the database).
        This is called in __init__ to capture the original values.
        """
        # Import drones database directly
        from drone_awe.drones import drones
        import copy
        
        # Find the drone in the database and make a deep copy
        drone_data = None
        for drone in drones:
            if drone['id'] == self.drone_name:
                drone_data = copy.deepcopy(drone)
                break
        
        if drone_data is None:
            raise ValueError(f"Drone '{self.drone_name}' not found in database")
        
        # Extract battery specs
        battery = drone_data.get('battery', {})
        capacity_mah = battery.get('batterycapacity', None)
        voltage_v = battery.get('batteryvoltage', None)
        
        # Always calculate energy from capacity and voltage
        if capacity_mah is not None and voltage_v is not None:
            energy_wh = (capacity_mah / 1000.0) * voltage_v
        else:
            energy_wh = battery.get('batteryenergy', None)
        
        return {
            'capacity_mah': capacity_mah,
            'voltage_v': voltage_v,
            'energy_wh': energy_wh,
            'charger_power_w': drone_data.get('chargerpowerrating', None),
            'recharge_time_min': drone_data.get('batteryrechargetime', None)
        }
    
    def _get_battery_specs(self):
        """
        Helper method to get battery specifications.
        Returns cached specs that were loaded in __init__ before any models could corrupt the database.
        
        Returns:
            Dictionary with battery specs: capacity_mah, voltage_v, energy_wh, charger_power_w, recharge_time_min
        """
        return self._battery_specs

    def _calculate_power(self, mission_speed, wind_speed, payload_mass, altitude=None):
        """
        Helper method to calculate power consumption for given flight conditions.
        
        Args:
            mission_speed: Horizontal speed in m/s (0 for hover)
            wind_speed: Wind speed in m/s
            payload_mass: Payload mass in kg
            altitude: Flight altitude in meters (uses cruise_altitude if None)
            
        Returns:
            Power consumption in Watts
        """
        if altitude is None:
            altitude = self.cruise_altitude
        
        model_input = {
            "dronename": self.drone_name,
            "windspeed": wind_speed,
            "wind": True if wind_speed > 0 else False,
            "winddirection": 0.0,  # Headwind (0 = headwind, 180 = tailwind)
            "mission": {
                "missionspeed": mission_speed,  # m/s
                "altitude": altitude,  # meters
                "heading": 0.0,
                "payload": payload_mass  # kg
            },
            "altitude": altitude,  # meters
            "temperature": 15.0,  # Celsius
            "xlabel": "missionspeed",  # Independent variable
            "ylabel": "power",  # Dependent variable (power consumption)
            "xvals": [mission_speed],  # Single value for this mission speed
            "zvals": [0.0],  # No z-variable iteration
            "simulationtype": "simple",
            "timestep": 1,
            "title": "Power Calculation"
        }
        
        # Create and run the model
        m = drone_awe.model(model_input, verbose=False)
        m.simulate()
        
        # Extract power consumption (in Watts)
        return m.classes['power'].params['power']

    def calculate_energy_cost(self, distance_meters, wind_speed, payload_mass, mission_speed=None, return_breakdown=False):
        """
        Uses drone-awe to estimate energy consumption for a complete flight leg,
        including takeoff/climb, cruise, and landing/descent phases.
        
        Args:
            distance_meters: Distance to travel in meters
            wind_speed: Wind speed in m/s
            payload_mass: Payload mass in kg
            mission_speed: Mission speed in m/s (optional, uses default if not provided)
            return_breakdown: If True, returns detailed breakdown of energy by phase
            
        Returns:
            If return_breakdown=False: Tuple of (total_energy_wh, total_flight_time_seconds)
            If return_breakdown=True: Tuple of (total_energy_wh, total_flight_time_seconds, breakdown_dict)
        """
        if mission_speed is None:
            mission_speed = self.default_speed
        
        # 1. Calculate hover power (for takeoff and landing)
        # Use a very small speed to approximate hover (0 m/s may cause issues in the model)
        hover_speed = 0.5  # m/s (near hover, avoids edge case at 0 m/s)
        hover_power = self._calculate_power(hover_speed, wind_speed, payload_mass, altitude=self.cruise_altitude)
        
        # 2. Calculate cruise power (forward flight at mission speed)
        cruise_power = self._calculate_power(mission_speed, wind_speed, payload_mass, altitude=self.cruise_altitude)
        
        # 3. Calculate climb time and energy
        climb_time = self.cruise_altitude / self.climb_rate  # seconds
        # Climb power is typically higher than hover due to additional work against gravity
        # Use a multiplier to account for climb power (typically 1.2-1.5x hover power)
        climb_power_multiplier = 1.3  # 30% more power during climb
        climb_power = hover_power * climb_power_multiplier
        climb_energy = (climb_power * climb_time) / 3600.0  # Wh
        
        # 4. Calculate cruise time and energy
        cruise_time = distance_meters / mission_speed  # seconds
        cruise_energy = (cruise_power * cruise_time) / 3600.0  # Wh
        
        # 5. Calculate descent time and energy
        descent_time = self.cruise_altitude / self.descent_rate  # seconds
        # Descent power is typically lower than hover (gravity assists)
        # Use a multiplier to account for descent power (typically 0.7-0.9x hover power)
        descent_power_multiplier = 0.8  # 20% less power during descent
        descent_power = hover_power * descent_power_multiplier
        descent_energy = (descent_power * descent_time) / 3600.0  # Wh
        
        # 6. Total energy and time
        total_energy_wh = climb_energy + cruise_energy + descent_energy
        total_flight_time = climb_time + cruise_time + descent_time
        
        if return_breakdown:
            breakdown = {
                'climb': {
                    'time': climb_time,
                    'power': climb_power,
                    'energy': climb_energy
                },
                'cruise': {
                    'time': cruise_time,
                    'power': cruise_power,
                    'energy': cruise_energy
                },
                'descent': {
                    'time': descent_time,
                    'power': descent_power,
                    'energy': descent_energy
                },
                'hover_power': hover_power
            }
            return total_energy_wh, total_flight_time, breakdown
        else:
            return total_energy_wh, total_flight_time

    def calculate_recharge_time(self, current_charge_percent=None, current_charge_wh=None, 
                               target_charge_percent=100.0, charging_efficiency=0.85):
        """
        Calculate the time required to recharge a drone battery.
        
        Args:
            current_charge_percent: Current battery charge as percentage (0-100)
            current_charge_wh: Current battery charge in Watt-hours (alternative to percent)
            target_charge_percent: Target charge level as percentage (default: 100%)
            charging_efficiency: Charging efficiency factor (default: 0.85, i.e., 85% efficient)
                                Accounts for energy loss during charging (heat, etc.)
            
        Returns:
            Tuple of (recharge_time_seconds, recharge_time_minutes, energy_to_charge_wh)
            
        Raises:
            ValueError: If battery specs are not available or charge values are invalid
        """
        # Get battery specifications
        specs = self._get_battery_specs()
        
        if specs['energy_wh'] is None:
            raise ValueError(f"Battery energy capacity not available for drone '{self.drone_name}'")
        
        total_energy_wh = specs['energy_wh']
        
        # Determine current charge in Wh
        if current_charge_wh is not None:
            current_wh = current_charge_wh
            current_percent = (current_wh / total_energy_wh) * 100.0
        elif current_charge_percent is not None:
            current_percent = current_charge_percent
            current_wh = (current_percent / 100.0) * total_energy_wh
        else:
            raise ValueError("Must provide either current_charge_percent or current_charge_wh")
        
        # Validate charge values
        if current_percent < 0 or current_percent > 100:
            raise ValueError(f"Current charge percentage must be between 0 and 100, got {current_percent}")
        if target_charge_percent < 0 or target_charge_percent > 100:
            raise ValueError(f"Target charge percentage must be between 0 and 100, got {target_charge_percent}")
        if current_percent >= target_charge_percent:
            return 0.0, 0.0, 0.0  # Already at or above target charge
        
        # Calculate energy needed to charge
        target_wh = (target_charge_percent / 100.0) * total_energy_wh
        energy_to_charge_wh = target_wh - current_wh
        
        # Get charger power rating
        charger_power_w = specs['charger_power_w']
        
        if charger_power_w is None:
            # Fallback: estimate from recharge time if available
            if specs['recharge_time_min'] is not None:
                # Estimate charger power from full recharge time
                # Assuming charging from 0% to 100% takes recharge_time_min
                # Energy needed = total_energy_wh / charging_efficiency
                estimated_charger_power_w = (total_energy_wh / charging_efficiency) / (specs['recharge_time_min'] / 60.0)
                charger_power_w = estimated_charger_power_w
            else:
                # Default fallback: use a reasonable default (e.g., 60W for typical drone charger)
                charger_power_w = 60.0
                import warnings
                warnings.warn(f"Charger power rating not available for '{self.drone_name}', using default {charger_power_w}W")
        
        # Calculate recharge time
        # Account for charging efficiency (not all input power goes to battery)
        # Time = Energy / (Charger Power * Efficiency)
        recharge_time_hours = energy_to_charge_wh / (charger_power_w * charging_efficiency)
        recharge_time_seconds = recharge_time_hours * 3600.0
        recharge_time_minutes = recharge_time_hours * 60.0
        
        return recharge_time_seconds, recharge_time_minutes, energy_to_charge_wh

if __name__ == "__main__":
    test_model = DronePhysicsModel()
    
    # Test energy cost calculation
    energy_used, flight_time_seconds, breakdown = test_model.calculate_energy_cost(1000, 5, 1, return_breakdown=True)
    print(f"Total energy used: {energy_used:.2f} Wh")
    print(f"Total flight time: {flight_time_seconds:.2f} seconds ({flight_time_seconds/60:.2f} minutes)")
    
    # Show breakdown
    print("\nEnergy breakdown:")
    print(f"  Climb ({breakdown['climb']['time']:.1f}s): {breakdown['climb']['energy']:.3f} Wh (power: {breakdown['climb']['power']:.1f} W)")
    print(f"  Cruise ({breakdown['cruise']['time']:.1f}s): {breakdown['cruise']['energy']:.3f} Wh (power: {breakdown['cruise']['power']:.1f} W)")
    print(f"  Descent ({breakdown['descent']['time']:.1f}s): {breakdown['descent']['energy']:.3f} Wh (power: {breakdown['descent']['power']:.1f} W)")
    print(f"\nHover power reference: {breakdown['hover_power']:.1f} W")
    
    # Test recharge time calculation
    print("\n" + "="*50)
    print("Recharge Time Calculation")
    print("="*50)
    
    # Get battery specs (already loaded in __init__)
    specs = test_model._get_battery_specs()
    print(f"\nBattery specifications for {test_model.drone_name}:")
    print(f"  Total capacity: {specs['energy_wh']:.2f} Wh")
    print(f"  Charger power: {specs['charger_power_w']:.1f} W" if specs['charger_power_w'] else "  Charger power: Not specified")
    
    # Test various charge levels
    test_cases = [
        (20.0, "20% charge"),
        (50.0, "50% charge"),
        (75.0, "75% charge"),
    ]
    
    for charge_percent, description in test_cases:
        recharge_sec, recharge_min, energy_needed = test_model.calculate_recharge_time(
            current_charge_percent=charge_percent,
            target_charge_percent=100.0
        )
        print(f"\nRecharging from {description} to 100%:")
        print(f"  Energy needed: {energy_needed:.2f} Wh")
        print(f"  Recharge time: {recharge_min:.1f} minutes ({recharge_sec:.0f} seconds)")
    
    # Test with energy in Wh
    print(f"\nRecharging after using {energy_used:.2f} Wh:")
    remaining_wh = max(0.0, specs['energy_wh'] - energy_used)  # Don't allow negative
    remaining_percent = (remaining_wh / specs['energy_wh']) * 100.0
    
    if remaining_wh > 0:
        recharge_sec, recharge_min, energy_needed = test_model.calculate_recharge_time(
            current_charge_wh=remaining_wh,
            target_charge_percent=100.0
        )
        print(f"  Remaining charge: {remaining_percent:.1f}% ({remaining_wh:.2f} Wh)")
        print(f"  Energy needed: {energy_needed:.2f} Wh")
        print(f"  Recharge time: {recharge_min:.1f} minutes ({recharge_sec:.0f} seconds)")
    else:
        # Battery would be completely drained (or over-discharged)
        print(f"  Warning: Flight used {energy_used:.2f} Wh, which exceeds battery capacity ({specs['energy_wh']:.2f} Wh)")
        print(f"  Battery would be completely drained. Full recharge needed:")
        recharge_sec, recharge_min, energy_needed = test_model.calculate_recharge_time(
            current_charge_percent=0.0,
            target_charge_percent=100.0
        )
        print(f"  Energy needed: {energy_needed:.2f} Wh")
        print(f"  Recharge time: {recharge_min:.1f} minutes ({recharge_sec:.0f} seconds)")