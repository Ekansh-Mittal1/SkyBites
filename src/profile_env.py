import time
import numpy as np
from env import SkyBitesEnv
from sb3_contrib.common.wrappers import ActionMasker

# 1. Setup
# Use the exact same config you intend to train with
env = SkyBitesEnv(
    orders_csv=None, # Use dynamic generation if you implemented it
    restaurants_json="config/restaurants.json", 
    pads_json="config/delivery_destinations.json",
    num_drones=10,
    dynamic_generation=True 
)

# Reset once to load everything
env.reset()

# 2. The Loop
N_STEPS = 5000
print(f"Running {N_STEPS} steps...")

start_time = time.time()

for _ in range(N_STEPS):
    # Sample a random action from the space
    action = env.action_space.sample()
    
    # Step the env
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        env.reset()

end_time = time.time()

# 3. Results
duration = end_time - start_time
sps = N_STEPS / duration

print(f"--------------------------------------------------")
print(f"Total Time: {duration:.2f} seconds")
print(f"Speed:      {sps:.2f} Steps Per Second (SPS)")
print(f"--------------------------------------------------")

# 4. Estimation
hours_for_1m = (1_000_000 / sps) / 3600
print(f"Estimated time for 1M steps: {hours_for_1m:.2f} hours")