from dotenv import load_dotenv
load_dotenv('azimuth.env')

import os
import sys
sys.path.append(os.getenv('PATH_ROOT', ''))

from VERICONF import conflict_checker
import numpy as np

# Define the Trajectory class for Python (matching the C++ struct)
class Trajectory:
    def __init__(self, waypoints, altitudes, speeds, cat):
        self.waypoints = waypoints  # List of (lat, lon) tuples
        self.altitudes = altitudes  # List of altitudes in feet
        self.speeds = speeds        # List of speeds in km/s
        self.cat = cat              # Wake turbulence category

def generate_random_trajectories(n_trajectories):
    # Define realistic bounds for the parameters
    LAT_BOUNDS = (30.0, 45.0)  # Continental US approximate bounds
    LON_BOUNDS = (-125.0, -70.0)
    ALT_BOUNDS = (20000, 40000)  # Typical cruise altitudes in feet
    SPEED_BOUNDS = (200, 500)    # Typical aircraft speeds in km/s
    WAYPOINTS_PER_TRAJ = 3       # Number of waypoints per trajectory

    trajectories = []
    
    for _ in range(n_trajectories):
        # Generate random waypoints with a general direction (west to east)
        start_lon = np.random.uniform(LON_BOUNDS[0], LON_BOUNDS[0] + 20)
        start_lat = np.random.uniform(*LAT_BOUNDS)
        
        waypoints = []
        for i in range(WAYPOINTS_PER_TRAJ):
            lon = start_lon + i * 5 + np.random.uniform(-2, 2)  # General west-to-east movement
            lat = start_lat + np.random.uniform(-2, 2)
            waypoints.append((lat, lon))
        
        # Generate ascending altitudes
        start_alt = np.random.uniform(*ALT_BOUNDS)
        altitudes = [
            start_alt + i * np.random.uniform(0, 1000)
            for i in range(WAYPOINTS_PER_TRAJ)
        ]
        
        # Generate reasonable speeds
        start_speed = np.random.uniform(*SPEED_BOUNDS)
        speeds = [
            start_speed + i * np.random.uniform(-10, 10)
            for i in range(WAYPOINTS_PER_TRAJ)
        ]
        
        # Random wake turbulence category (1-4)
        cat = np.random.randint(1, 5)
        
        traj = Trajectory(waypoints, altitudes, speeds, cat)
        trajectories.append(traj)
    
    return trajectories

# Generate 1000 trajectories
trajectories = generate_random_trajectories(1000)

# Convert to C++ Trajectory structs
cpp_trajectories = []
for traj in trajectories:
    cpp_traj = conflict_checker.Trajectory()
    cpp_traj.waypoints = traj.waypoints
    cpp_traj.altitudes = traj.altitudes
    cpp_traj.speeds = traj.speeds
    cpp_traj.cat = traj.cat
    cpp_trajectories.append(cpp_traj)


# Check for conflicts
import time

start_time = time.time()
conflicts = conflict_checker.check_conflicts(cpp_trajectories)
end_time = time.time()


# Display conflicts
for conflict in conflicts:
    traj1_idx, traj2_idx, seg1_idx, seg2_idx = conflict
    print(f"Conflict between Trajectory {traj1_idx} Segment {seg1_idx} and Trajectory {traj2_idx} Segment {seg2_idx}")

print(f"Conflict checking time: {(end_time - start_time) * 1000:.4f} ms")
