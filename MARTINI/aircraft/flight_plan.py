from typing import List, Tuple
import numpy as np
from . import trajectory_simple as ts

class FlightPlan:
    """
    A class to represent an aircraft flight plan with waypoints, altitudes, and speeds.
    """
    def __init__(self, 
                 waypoints: List[Tuple[float, float]], 
                 altitudes: List[float], 
                 speeds: List[float],
                 entry_time: float = 0,
                 entry_psi: float = 0):
        """
        Initialize a flight plan.
        
        Args:
            waypoints: List of (x, y) coordinates for each waypoint
            altitudes: List of altitudes (ft) for each waypoint
            speeds: List of speeds (knots) for each waypoint
        """
        # Validate input lengths match
        if not (len(waypoints) == len(altitudes) == len(speeds)):
            raise ValueError("Waypoints, altitudes, and speeds must have the same length")
        
        self.waypoints = np.array(waypoints)
        self.altitudes = np.array(altitudes)
        self.speeds = np.array(speeds)
        self.entry_time = entry_time
        self.entry_psi = entry_psi
        
    @property
    def num_waypoints(self) -> int:
        """Return the number of waypoints in the flight plan."""
        return len(self.waypoints)
    
    def get_waypoint(self, index: int) -> Tuple[Tuple[float, float], float, float]:
        """
        Get the complete information for a waypoint at given index.
        
        Returns:
            Tuple containing ((x, y), altitude, speed)
        """
        return (tuple(self.waypoints[index]), 
                self.altitudes[index], 
                self.speeds[index])
    
    def __len__(self) -> int:
        """Return the number of waypoints in the flight plan."""
        return self.num_waypoints

    def get_trajectory_4d(self) -> List[np.ndarray]:
        return ts.get_4d_trajectory_from_flight_plan(
            self.entry_time,
            self.entry_psi,
            self.speeds,
            self.altitudes,
            self.waypoints
        )
    
    def __str__(self):
        str_repr = f"Flight plan with {self.num_waypoints} waypoints"
        for i, wp in enumerate(self.waypoints):
            str_repr += f"\n* Waypoint {i}: {wp}"
            str_repr += f"\nAltitude: {self.altitudes[i]}"
            str_repr += f"\nSpeed: {self.speeds[i]}"
        return str_repr
    