import numpy as np
import json
import os

class DronePhysicsModel:
    def __init__(self, drone_config="skyranger_r70"):
        """
        Pure physics implementation using Momentum Theory & Aerodynamic Drag.
        No external libraries required.
        
        Args:
            drone_config: Name of the drone configuration file (without .json extension)
                         Config files should be in the config/ directory
        """
        # Load configuration from JSON file
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            f"{drone_config}.json"
        )
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Drone configuration file not found: {config_path}\n"
                f"Please ensure the config file exists in the config/ directory."
            )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 1. PHYSICAL CONSTANTS
        physical = config.get("physical_constants", {})
        self.g = physical.get("gravity", 9.81)
        self.rho = physical.get("air_density", 1.225)
        self.eta = physical.get("propulsive_efficiency", 0.75)
        
        # 2. DRONE SPECIFICATIONS
        # Weight
        weight = config.get("weight", {})
        self.mass_empty = weight.get("mass_empty_kg", 4.7)
        self.mass_max = weight.get("mass_max_kg", 8.5)
        
        # Aerodynamics
        aero = config.get("aerodynamics", {})
        self.cd = aero.get("drag_coefficient", 0.6)
        self.area_front = aero.get("frontal_area_m2", 0.15)
        
        # Rotors (Lift System)
        rotors = config.get("rotors", {})
        self.num_rotors = rotors.get("num_rotors", 4)
        self.rotor_diameter = rotors.get("rotor_diameter_m", 0.55)
        self.rotor_radius = self.rotor_diameter / 2
        # Total Disk Area = N * pi * r^2
        self.area_disk = self.num_rotors * np.pi * (self.rotor_radius ** 2)
        
        # Battery (Energy Source)
        battery = config.get("battery", {})
        self.battery_capacity_wh = battery.get("capacity_wh", 333.0)
        self.charger_power_w = battery.get("charger_power_w", 200.0)
        self.battery_voltage_v = battery.get("voltage_v", 22.2)
        
        # Flight Limits
        limits = config.get("flight_limits", {})
        self.speed_limit = limits.get("speed_limit_ms", 15.0)
        self.climb_rate = limits.get("climb_rate_ms", 3.0)
        self.descent_rate = limits.get("descent_rate_ms", 2.0)
        self.default_speed = limits.get("default_speed_ms", 10.0)
        
        # Avionics
        avionics = config.get("avionics", {})
        self.p_avionics = avionics.get("base_power_w", 15.0)

    def _calculate_power(self, speed_ms, payload_kg, wind_speed_ms=0.0):
        """
        Calculates instantaneous power (Watts) required for steady flight.
        Uses Momentum Theory for lift and Drag Equation for forward motion.
        """
        # A. Total Mass & Thrust
        total_mass = self.mass_empty + payload_kg
        thrust_newtons = total_mass * self.g
        
        # B. Effective Airspeed (Ground Speed + Wind Component)
        # We assume headwind for worst-case energy estimation
        airspeed = speed_ms + wind_speed_ms
        
        # C. Induced Power (Lift)
        # P_induced = T^1.5 / sqrt(2 * rho * A)
        # Note: Induced power actually drops slightly with forward speed, 
        # but using the hover value is a safe, robust conservative estimate for RL.
        p_induced_ideal = (thrust_newtons ** 1.5) / np.sqrt(2 * self.rho * self.area_disk)
        
        # D. Parasitic Power (Drag)
        # P_drag = 0.5 * Cd * rho * A * v^3
        p_parasitic = 0.5 * self.cd * self.rho * self.area_front * (airspeed ** 3)
        
        # E. Total Power (Watts)
        # Scale by efficiency factor (eta)
        p_total = (p_induced_ideal + p_parasitic) / self.eta
        
        # F. Avionics Base Load (Computers, Sensors, Comms)
        return p_total + self.p_avionics

    def calculate_energy_cost(self, distance_meters, wind_speed, payload_mass, mission_speed=None, return_breakdown=False):
        """
        Integrates power over time to find total energy cost (Wh) for a flight leg.
        """
        if mission_speed is None: mission_speed = self.default_speed
        
        # 1. PHASE: CLIMB (Takeoff to Cruise Altitude)
        # Assume 100m cruise altitude
        cruise_altitude = 100.0
        climb_time = cruise_altitude / self.climb_rate
        # Climb requires extra power (lifting work). Approx 1.3x hover power.
        p_hover = self._calculate_power(0.0, payload_mass, wind_speed)
        p_climb = p_hover * 1.2
        e_climb_wh = (p_climb * climb_time) / 3600.0
        
        # 2. PHASE: CRUISE (Travel)
        cruise_time = distance_meters / mission_speed
        p_cruise = self._calculate_power(mission_speed, payload_mass, wind_speed)
        e_cruise_wh = (p_cruise * cruise_time) / 3600.0
        
        # 3. PHASE: DESCENT (Landing)
        descent_time = cruise_altitude / self.descent_rate
        # Descent uses less power. Approx 0.8x hover power.
        p_descent = p_hover * 0.8
        e_descent_wh = (p_descent * descent_time) / 3600.0
        
        # 4. TOTALS
        total_energy_wh = e_climb_wh + e_cruise_wh + e_descent_wh
        total_time_s = climb_time + cruise_time + descent_time
        
        if return_breakdown:
            return total_energy_wh, total_time_s, {
                "hover_power": p_hover,
                "cruise_power": p_cruise,
                "climb_energy": e_climb_wh,
                "cruise_energy": e_cruise_wh
            }
        
        return total_energy_wh, total_time_s

    def calculate_recharge_time(self, current_charge_wh, target_charge_percent=100.0, charging_efficiency=0.9):
        """
        Calculates time to charge battery.
        """
        target_wh = self.battery_capacity_wh * (target_charge_percent / 100.0)
        needed_wh = max(0.0, target_wh - current_charge_wh)
        
        # Time = Energy / (Power * Efficiency)
        hours = needed_wh / (self.charger_power_w * charging_efficiency)
        
        return hours * 3600, hours * 60, needed_wh
    
    def _get_battery_specs(self):
        """Helper for environment compatibility"""
        return {
            'energy_wh': self.battery_capacity_wh,
            'capacity_mah': (self.battery_capacity_wh / self.battery_voltage_v) * 1000,
            'voltage_v': self.battery_voltage_v
        }

if __name__ == "__main__":
    # VALIDATION
    model = DronePhysicsModel()
    print(f"--- Physics Model Validation (SkyRanger R70 Config) ---")
    print(f"Empty Mass: {model.mass_empty} kg")
    print(f"Max Mass:   {model.mass_max} kg")
    print(f"Battery:    {model.battery_capacity_wh} Wh")
    
    # Test 1: Hover (0 m/s) with 1kg Payload
    p_hover = model._calculate_power(0, 1.0)
    print(f"\nHover Power (1kg payload): {p_hover:.1f} Watts")
    # Expected: ~800-1000W for a 5.7kg drone
    
    # Test 2: Cruise (15 m/s) with 1kg Payload
    p_cruise = model._calculate_power(15.0, 1.0)
    print(f"Cruise Power (15 m/s):     {p_cruise:.1f} Watts")
    
    # Test 3: Full Mission (1km trip)
    e, t, b = model.calculate_energy_cost(1000, 5.0, 1.0, return_breakdown=True)
    print(f"\n1km Mission Cost (1kg payload, 5m/s wind):")
    print(f"  Time:   {t:.1f} seconds")
    print(f"  Energy: {e:.2f} Wh")
    print(f"  % Batt: {(e / model.battery_capacity_wh)*100:.1f}%")