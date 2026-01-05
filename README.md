# SkyBites: Reinforcement Learning for Drone Delivery Vehicle Routing Problem

[View our final paper here](https://drive.google.com/file/d/1kELwjyp5n3pmg8QKiyyxayAfrYWAhnqL/view)

SkyBites is a reinforcement learning (RL) system for optimizing drone fleet operations in a food delivery scenario. The project uses Proximal Policy Optimization (PPO) with action masking to train agents that coordinate multiple drones to pick up orders from restaurants and deliver them to customer locations while managing battery constraints and charging schedules.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
  - [Order Simulation](#order-simulation)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Environment Details](#environment-details)
- [Training Details](#training-details)
- [Troubleshooting](#troubleshooting)

## Overview

SkyBites simulates a realistic drone delivery system where:
- Multiple drones operate from a central base station
- Orders arrive dynamically from restaurants throughout the day
- Drones must pick up orders from restaurants and deliver them to customer locations
- Drones have limited battery capacity and must return to base for charging
- The RL agent learns to optimize drone assignments to maximize order fulfillment while managing energy constraints

The system uses **MaskablePPO** from Stable-Baselines3 to train policies that respect physical constraints (e.g., drones can't be reassigned while in flight) through action masking.

## Features

- **Realistic Physics Model**: Energy consumption calculations based on momentum theory and aerodynamic drag
- **Dynamic Order Generation**: Orders are generated on-the-fly during training for better generalization
- **Action Masking**: Prevents invalid actions (e.g., assigning tasks to drones that are already moving)
- **Automatic Order Chaining**: Optional mode where drones automatically route restaurant → dropoff → base
- **Parallel Training**: Uses multiple CPU cores for faster data collection
- **Comprehensive Evaluation**: Detailed logging of order assignments, completions, and failures
- **Visualization**: Generate videos showing drone movements and order fulfillment

## Architecture

### Core Components

1. **Environment (`src/env.py`)**: Gymnasium-compatible environment that simulates the drone delivery system
   - Uses SimPy for discrete-event simulation of continuous-time processes
   - Integrates with physics model for energy calculations
   - Provides observations and rewards to the RL agent

2. **Physics Model (`src/drone_physics.py`)**: Calculates energy consumption and flight times
   - Based on momentum theory for lift and drag equations for forward motion
   - Accounts for payload, wind, climb/cruise/descent phases
   - Configurable via JSON files (e.g., `config/skyranger_r70.json`)

3. **Training Script (`src/train.py`)**: Trains MaskablePPO agents
   - Supports parallel environments for faster training
   - Automatic checkpointing and evaluation
   - TensorBoard logging

4. **Evaluation Script (`src/evaluate.py`)**: Evaluates trained models
   - Runs multiple episodes and collects detailed statistics
   - Logs order assignments, completions, and failures
   - Generates summary reports

5. **Visualization Script (`src/visualize.py`)**: Creates video visualizations
   - Shows drone positions, orders, and map background
   - Generates MP4 videos of policy execution

6. **Order Simulation (`src/simulate_orders.py`)**: Generates order data
   - Uses Poisson processes based on restaurant operating hours
   - Assigns delivery locations proportionally
   - Can generate static CSV files or dynamic in-memory data

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Final
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `gymnasium`: RL environment interface
- `simpy`: Discrete-event simulation
- `stable-baselines3`: RL algorithms
- `sb3-contrib`: MaskablePPO implementation
- `pandas`, `numpy`: Data processing
- `geopy`: Geographic distance calculations
- `matplotlib`, `contextily`: Visualization
- `imageio`, `imageio-ffmpeg`: Video generation
- `tensorboard`: Training monitoring

## Configuration

### Configuration Files

The project uses JSON configuration files in the `config/` directory:

- **`restaurants.json`**: Restaurant locations, operating hours, and order rates
- **`delivery_destinations.json`**: Customer delivery locations with proportional weights
- **`base.json`**: Base station location for drone charging
- **`skyranger_r70.json`**: Drone physical specifications (mass, battery, aerodynamics, etc.)

### Example Configuration

See `config/restaurants.json` for restaurant configuration format:
```json
{
  "restaurant_id": "R001",
  "name": "The Axe and Palm",
  "operating_hours": {
    "start_hour": 11,
    "end_hour": 2
  },
  "hourly_order_rates": {
    "11": 3.0,
    "12": 5.0,
    ...
  },
  "location": {
    "latitude": 37.4249,
    "longitude": -122.1708
  }
}
```

## Usage

### Training

Train a new model with default settings:
```bash
python src/train.py
```

Train with custom parameters:
```bash
python src/train.py \
    --num-drones 10 \
    --total-timesteps 5000000 \
    --learning-rate 3e-4 \
    --run-name PPO_10_chained_20 \
    --auto-chain-orders \
    --order-scale-factor 20
```

**Key Training Arguments:**
- `--num-drones`: Number of drones in the fleet (default: 10)
- `--total-timesteps`: Total training steps (default: 5000000)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--run-name`: Custom name for this training run
- `--auto-chain-orders`: Enable automatic routing (restaurant → dropoff → base)
- `--order-scale-factor`: Scale down orders (20 = 1/20 of orders, for faster training)
- `--n-steps`: Steps per update (default: 4096)
- `--batch-size`: Batch size (default: 256)
- `--gamma`: Discount factor (default: 0.999)

**Training Output:**
- Models saved to `models/{run_name}/`
- TensorBoard logs in `logs/tensorboard/{run_name}/`
- Best model: `models/{run_name}/best_model/best_model.zip`
- Final model: `models/{run_name}/{run_name}_final.zip`
- Normalization stats: `models/{run_name}/{run_name}_vec_normalize.pkl`

**Monitor Training:**
```bash
tensorboard --logdir logs/tensorboard
```

### Evaluation

Evaluate a trained model:
```bash
python src/evaluate.py PPO_10_chained_20 --episodes 10
```

**Evaluation Arguments:**
- `run_name`: Name of the training run to evaluate
- `--episodes`: Number of evaluation episodes (default: 5)
- `--num-drones`: Number of drones (must match training)
- `--auto-chain-orders`: Enable if used during training
- `--order-scale-factor`: Must match training value

**Evaluation Output:**
- Detailed logs: `outs/eval_log_{run_name}_{timestamp}.txt`
- Summary report: `outs/eval_summary_{run_name}_{timestamp}.txt`

### Visualization

Generate a video visualization of a trained policy:
```bash
python src/visualize.py PPO_10_chained_20 --fps 10
```

**Visualization Arguments:**
- `run_name`: Name of the training run
- `--output`: Custom output path (default: `outs/visualization_{run_name}.mp4`)
- `--fps`: Frames per second (default: 10)
- `--auto-chain-orders`: Enable if used during training
- `--order-scale-factor`: Must match training value

### Order Simulation

Generate a static CSV file of orders:
```bash
python src/simulate_orders.py \
    config/restaurants.json \
    -o orders.csv \
    --delivery-destinations config/delivery_destinations.json \
    -d 24
```

**Arguments:**
- `config`: Path to restaurants JSON (default: `config/restaurants.json`)
- `-o, --output`: Output CSV path (default: `orders.csv`)
- `--delivery-destinations`: Delivery destinations JSON
- `-d, --duration`: Simulation duration in hours (default: 24)

## Project Structure

```
Final/
├── config/                 # Configuration files
│   ├── base.json          # Base station location
│   ├── restaurants.json   # Restaurant configurations
│   ├── delivery_destinations.json  # Delivery locations
│   └── skyranger_r70.json # Drone physics specs
├── src/                    # Source code
│   ├── env.py             # Main environment
│   ├── drone_physics.py   # Physics calculations
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── visualize.py       # Visualization script
│   ├── simulate_orders.py # Order generation
│   ├── debug_data.py      # Data validation utility
│   └── profile_env.py     # Performance profiling
├── models/                 # Trained models
│   └── {run_name}/        # Per-run model directories
├── logs/                   # Training logs
│   └── tensorboard/       # TensorBoard event files
├── outs/                   # Evaluation and visualization outputs
├── orders.csv             # Generated order data (optional)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── TRAINING.md            # Detailed training guide
```

## Key Components

### Environment (`SkyBitesEnv`)

**Observation Space:**
- `fleet`: Array of shape `(num_drones, 5)` containing for each drone:
  - Latitude, Longitude
  - Battery level (Wh)
  - Status (IDLE, MOVING, SERVICE, CHARGING, DEAD)
  - Payload (kg)
- `orders`: Array of shape `(MAX_ORDERS_IN_OBS, 3)` containing for each order:
  - Order age (minutes)
  - Pickup location ID (restaurant node ID)
  - Dropoff location ID (pad node ID)

**Action Space:**
- `MultiDiscrete([num_nodes] * num_drones)`
- Each drone selects a target node (restaurant, pad, or base)
- Actions are masked to prevent invalid assignments

**Reward Function:**
- +100 for each order assigned to a drone
- -1 per minute for late orders (age > 30 min, capped at -20 per order)
- -100 for each crashed drone (one-time penalty)
- -5 for each failed order (insufficient battery)

**Episode Termination:**
- Episode ends after 24 hours (1440 minutes)

### Action Masking (`DroneActionMaskWrapper`)

The wrapper implements `action_masks()` to prevent invalid actions:
- **Blocked**: Drones that are MOVING, SERVICING, CHARGING (in chaining mode), or DEAD
- **Allowed**: Drones that are IDLE or CHARGING can be assigned tasks
- **Physics Check**: Only allows restaurant assignments if the drone can complete the full mission (restaurant → dropoff → base)

### Physics Model (`DronePhysicsModel`)

Calculates energy consumption using:
- **Momentum Theory**: For lift/thrust calculations
- **Drag Equation**: For forward motion resistance
- **Flight Phases**: Separate calculations for climb, cruise, and descent

Energy depends on:
- Distance traveled
- Payload mass
- Wind speed (optional)
- Flight speed

### Order Generation

Orders are generated using:
- **Poisson Process**: Based on hourly order rates per restaurant
- **Operating Hours**: Restaurants have configurable start/end hours
- **Delivery Assignment**: Proportional assignment to delivery destinations

## Environment Details

### Drone States

- **IDLE**: At base or a location, ready for assignment
- **MOVING**: In flight to a target location
- **SERVICE**: Picking up or dropping off an order (30 seconds)
- **CHARGING**: At base, recharging battery
- **DEAD**: Crashed due to insufficient battery

### Order Lifecycle

1. **Spawn**: Order appears at restaurant at scheduled time
2. **Assignment**: Agent assigns a drone to the restaurant
3. **Pickup**: Drone arrives and picks up order (30 seconds)
4. **Delivery**: Drone flies to delivery location and drops off (30 seconds)
5. **Completion**: Order is removed from active queue

### Battery Management

- Drones start with full battery at base
- Energy consumed during flight based on physics model
- Drones can charge at base (rate depends on charger power)
- Drones crash if battery depletes during flight
- Agent must plan routes to ensure sufficient battery for return to base

## Training Details

### Algorithm

- **MaskablePPO** from `sb3-contrib`
- **Policy Network**: Multi-input policy (handles Dict observation space)
- **Network Architecture**: `[256, 256]` hidden layers
- **Normalization**: VecNormalize for reward normalization (observation normalization disabled)

### Parallelization

- Uses `SubprocVecEnv` for parallel data collection
- Number of parallel environments = CPU count
- Steps per update are divided across environments

### Hyperparameters (Default)

- Learning rate: `3e-4`
- Steps per update: `4096` (total, divided by num_envs)
- Batch size: `256`
- Discount factor (gamma): `0.999`
- Epochs per update: `10` (PPO default)

### Training Tips

1. **Start Small**: Begin with fewer timesteps (e.g., 100k) to test setup
2. **Monitor Early**: Check TensorBoard after a few thousand steps
3. **Use Order Scaling**: Set `--order-scale-factor 20` for faster training
4. **Enable Auto-Chaining**: Use `--auto-chain-orders` to simplify action space
5. **Adjust Fleet Size**: Fewer drones = faster training, more drones = more complex coordination

## Troubleshooting

### Import Errors

Ensure all packages are installed:
```bash
pip install -r requirements.txt
```

### Out of Memory

- Reduce `--n-steps` (e.g., 2048 instead of 4096)
- Reduce `--batch-size` (e.g., 128 instead of 256)
- Reduce `--num-drones`

### Slow Training

- Reduce `--eval-freq` to evaluate less frequently
- Use `--order-scale-factor` to reduce number of orders
- Reduce number of parallel environments (modify `train.py`)

### Model Not Learning

- Check TensorBoard for reward trends
- Verify action masking is working (should see valid actions only)
- Ensure reward normalization is enabled
- Try adjusting learning rate or network architecture

### Evaluation Issues

- Ensure `--auto-chain-orders` and `--order-scale-factor` match training settings
- Check that normalization stats file exists and loads correctly
- Verify model file path is correct

### Visualization Issues

- Install `imageio` and `imageio-ffmpeg` for video generation
- Check that model and stats files exist
- Ensure matplotlib backend supports non-interactive mode

## License

[Add license information if applicable]

## Citation

[Add citation information if applicable]
