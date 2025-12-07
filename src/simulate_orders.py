#!/usr/bin/env python3
"""
Order Simulation Script for RL-Based Drone VRP

This script simulates restaurant orders based on restaurant configurations,
operating hours, and average orders per hour. Outputs CSV format optimized
for RL training.
"""

import json
import csv
import argparse
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any


def load_restaurant_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Load restaurant configuration from JSON file.
    
    Expected schema:
    [
        {
            "restaurant_id": str or int,
            "name": str,
            "operating_hours": {
                "start_hour": int (0-23),
                "end_hour": int (0-23)
            },
            "hourly_order_rates": {
                "0": float,  # orders per hour for hour 0 (midnight-1am)
                "1": float,  # orders per hour for hour 1 (1am-2am)
                ...
                "23": float  # orders per hour for hour 23 (11pm-midnight)
            }
            OR
            "avg_orders_per_hour": float  # fallback if hourly_order_rates not provided
            "location": {
                "latitude": float,
                "longitude": float
            }
        },
        ...
    ]
    
    Schema is extensible - additional fields can be added without breaking
    existing code (e.g., order_value, prep_time).
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_delivery_destinations(config_path: str) -> List[Dict[str, Any]]:
    """
    Load delivery destinations configuration from JSON file.
    
    Expected schema:
    [
        {
            "destination_id": str,
            "name": str,
            "location": {
                "latitude": float,
                "longitude": float
            },
            "proportion": float  # relative weight (will be normalized to sum to 1.0)
        },
        ...
    ]
    
    Proportions are automatically normalized so they sum to 1.0.
    """
    with open(config_path, 'r') as f:
        destinations = json.load(f)
    
    # Normalize proportions so they sum to 1.0
    total_proportion = sum(d['proportion'] for d in destinations)
    if total_proportion > 0:
        for destination in destinations:
            destination['proportion'] = destination['proportion'] / total_proportion
    else:
        # If all proportions are 0, set equal weights
        equal_weight = 1.0 / len(destinations) if destinations else 1.0
        for destination in destinations:
            destination['proportion'] = equal_weight
    
    return destinations


def assign_delivery_location(destinations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assign a delivery location to an order proportionally based on destination proportions.
    
    Args:
        destinations: List of destination dictionaries with 'proportion' field
        
    Returns:
        Selected destination dictionary
    """
    destination_ids = [d['destination_id'] for d in destinations]
    proportions = [d['proportion'] for d in destinations]
    
    # Normalize proportions (in case of floating point errors)
    total = sum(proportions)
    proportions = [p / total for p in proportions]
    
    # Select destination using weighted random choice
    selected_idx = np.random.choice(len(destinations), p=proportions)
    return destinations[selected_idx]


def generate_orders_for_restaurant(
    restaurant: Dict[str, Any],
    start_time: datetime,
    duration_hours: int,
    order_id_counter: int,
    delivery_destinations: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate orders for a single restaurant using Poisson process.
    
    Args:
        restaurant: Restaurant configuration dictionary
        start_time: Simulation start datetime
        duration_hours: Total simulation duration in hours
        order_id_counter: Starting order ID counter
        delivery_destinations: Optional list of delivery destinations for assignment
        
    Returns:
        List of order dictionaries and updated order_id_counter
    """
    orders = []
    restaurant_id = restaurant['restaurant_id']
    restaurant_name = restaurant['name']
    start_hour = restaurant['operating_hours']['start_hour']
    end_hour = restaurant['operating_hours']['end_hour']
    
    # Extract restaurant location
    restaurant_location = restaurant.get('location', None)
    restaurant_lat = restaurant_location['latitude'] if restaurant_location else None
    restaurant_lon = restaurant_location['longitude'] if restaurant_location else None
    
    # Get hourly order rates if available, otherwise use single avg_orders_per_hour
    hourly_rates = restaurant.get('hourly_order_rates', None)
    if hourly_rates is None:
        # Fallback to single average rate for backward compatibility
        avg_orders_per_hour = restaurant.get('avg_orders_per_hour', 0.0)
        # Convert to hourly_rates dict for uniform processing
        hourly_rates = {str(h): avg_orders_per_hour for h in range(24)}
    
    # Convert string keys to int keys for easier processing
    hourly_rates = {int(k): float(v) for k, v in hourly_rates.items()}
    
    # Calculate operating hours (handle wrap-around if end < start)
    if end_hour <= start_hour:
        # Restaurant operates overnight (e.g., 22:00 to 6:00)
        operating_hours = list(range(start_hour, 24)) + list(range(0, end_hour))
    else:
        operating_hours = list(range(start_hour, end_hour))
    
    current_order_id = order_id_counter
    
    # Generate orders for each hour in the simulation
    for hour_offset in range(duration_hours):
        current_hour = (start_time.hour + hour_offset) % 24
        
        # Check if restaurant is open during this hour
        if current_hour not in operating_hours:
            continue
        
        # Get order rate for this specific hour
        order_rate = hourly_rates.get(current_hour, 0.0)
        
        # Skip if no orders expected for this hour
        if order_rate <= 0:
            continue
        
        # Generate number of orders for this hour using Poisson distribution
        num_orders = np.random.poisson(order_rate)
        
        # Generate timestamps within this hour (in minutes)
        for _ in range(num_orders):
            # Random minute within the hour (0-59)
            minute_offset = np.random.randint(0, 60)
            
            # Calculate absolute timestamp in minutes from simulation start
            total_minutes = hour_offset * 60 + minute_offset
            
            # Assign delivery location proportionally
            delivery_dest = None
            delivery_lat = None
            delivery_lon = None
            delivery_location_id = None
            delivery_location_name = None
            
            if delivery_destinations:
                delivery_dest = assign_delivery_location(delivery_destinations)
                delivery_lat = delivery_dest['location']['latitude']
                delivery_lon = delivery_dest['location']['longitude']
                delivery_location_id = delivery_dest['destination_id']
                delivery_location_name = delivery_dest['name']
            
            order = {
                'order_id': f'ORD_{current_order_id:06d}',
                'restaurant_id': str(restaurant_id),
                'restaurant_name': restaurant_name,
                'timestamp_minutes': total_minutes,
                'hour_of_day': current_hour,
                'restaurant_latitude': restaurant_lat,
                'restaurant_longitude': restaurant_lon,
                'restaurant_location_id': str(restaurant_id),
                'delivery_latitude': delivery_lat,
                'delivery_longitude': delivery_lon,
                'delivery_location_id': delivery_location_id,
                'delivery_location_name': delivery_location_name
            }
            
            orders.append(order)
            current_order_id += 1
    
    return orders, current_order_id


def simulate_orders(
    config_path: str,
    output_path: str,
    start_time: datetime = None,
    duration_hours: int = 24,
    delivery_destinations_path: str = None
) -> None:
    """
    Main simulation function.
    
    Args:
        config_path: Path to restaurant configuration JSON file
        output_path: Path to output CSV file
        start_time: Simulation start time (default: current time)
        duration_hours: Simulation duration in hours (default: 24)
        delivery_destinations_path: Optional path to delivery destinations JSON file
    """
    # Load restaurant configurations
    restaurants = load_restaurant_config(config_path)
    
    # Load delivery destinations if provided
    delivery_destinations = None
    if delivery_destinations_path:
        try:
            delivery_destinations = load_delivery_destinations(delivery_destinations_path)
            print(f"Loaded {len(delivery_destinations)} delivery destinations")
        except FileNotFoundError:
            print(f"Warning: Delivery destinations file not found: {delivery_destinations_path}")
            print("Continuing without delivery location assignment")
    
    if start_time is None:
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Generate orders for all restaurants
    all_orders = []
    order_id_counter = 1
    
    for restaurant in restaurants:
        orders, order_id_counter = generate_orders_for_restaurant(
            restaurant, start_time, duration_hours, order_id_counter, delivery_destinations
        )
        all_orders.extend(orders)
    
    # Sort orders by timestamp for sequential RL training
    all_orders.sort(key=lambda x: x['timestamp_minutes'])
    
    # Write to CSV
    fieldnames = [
        'order_id', 'restaurant_id', 'restaurant_name', 
        'timestamp_minutes', 'hour_of_day',
        'restaurant_latitude', 'restaurant_longitude', 'restaurant_location_id',
        'delivery_latitude', 'delivery_longitude', 'delivery_location_id', 'delivery_location_name'
    ]
    
    # Ensure fieldnames are clean (strip any accidental spaces)
    fieldnames = [f.strip() for f in fieldnames]
    
    # Also ensure order dictionary keys are clean (strip spaces from keys)
    cleaned_orders = []
    for order in all_orders:
        cleaned_order = {k.strip(): v for k, v in order.items()}
        cleaned_orders.append(cleaned_order)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_orders)
    
    print(f"Generated {len(all_orders)} orders")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Simulate restaurant orders for RL-based drone VRP training'
    )
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='config/restaurants.json',
        help='Path to restaurant configuration JSON file (default: config/restaurants.json)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='orders.csv',
        help='Output CSV file path (default: orders.csv)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=24,
        help='Simulation duration in hours (default: 24)'
    )
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Simulation start time (YYYY-MM-DD HH:MM, default: current time)'
    )
    parser.add_argument(
        '--delivery-destinations',
        type=str,
        default='config/delivery_destinations.json',
        help='Path to delivery destinations JSON file (default: config/delivery_destinations.json)'
    )
    
    args = parser.parse_args()
    
    start_time = None
    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M')
    
    delivery_dest_path = args.delivery_destinations
    
    simulate_orders(
        config_path=args.config,
        output_path=args.output,
        start_time=start_time,
        duration_hours=args.duration,
        delivery_destinations_path=delivery_dest_path
    )


def generate_random_day(restaurants_config, pads_config, duration_hours=24):
    """
    Generates a DataFrame of orders in-memory for a single episode.
    
    Args:
        restaurants_config: List of restaurant configuration dictionaries
        pads_config: List of delivery destination configuration dictionaries
        duration_hours: Duration of the simulation in hours (default: 24)
    
    Returns:
        pandas.DataFrame with columns: order_id, restaurant_id, timestamp_minutes,
        delivery_location_id, delivery_latitude, delivery_longitude,
        restaurant_latitude, restaurant_longitude
    """
    # Normalize delivery destination proportions if needed
    # (reuse the normalization logic from load_delivery_destinations)
    # Make a copy to avoid modifying the original config
    if pads_config:
        pads_config = copy.deepcopy(pads_config)
        total_proportion = sum(d.get('proportion', 0) for d in pads_config)
        if total_proportion > 0:
            for d in pads_config:
                d['proportion'] = d.get('proportion', 0) / total_proportion
        else:
            # If all proportions are 0, set equal weights
            equal_weight = 1.0 / len(pads_config) if pads_config else 1.0
            for d in pads_config:
                d['proportion'] = equal_weight
    
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_orders = []
    order_id_counter = 1
    
    for restaurant in restaurants_config:
        orders, order_id_counter = generate_orders_for_restaurant(
            restaurant, 
            start_time, 
            duration_hours, 
            order_id_counter, 
            pads_config
        )
        all_orders.extend(orders)
    
    # Create DataFrame directly
    if not all_orders:
        return pd.DataFrame(columns=[
            'order_id', 'restaurant_id', 'timestamp_minutes', 
            'delivery_location_id', 'delivery_latitude', 'delivery_longitude',
            'restaurant_latitude', 'restaurant_longitude'
        ])
    
    df = pd.DataFrame(all_orders)
    return df.sort_values('timestamp_minutes')


if __name__ == '__main__':
    main()

