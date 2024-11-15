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
    
    def plot(self):
        """Plot the flight plan waypoints and trajectory."""
        import matplotlib.pyplot as plt

        # Plot waypoints
        plt.plot(self.waypoints[:,0], self.waypoints[:,1], 'bo-', label='Flight Path')
        
        # Add waypoint labels
        for i, wp in enumerate(self.waypoints):
            plt.annotate(f'WP{i}\n{self.altitudes[i]}ft\n{self.speeds[i]}kts', 
                        (wp[0], wp[1]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

        plt.xlabel('X Position')
        plt.ylabel('Y Position') 
        plt.title('Flight Plan')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

    def validate_no_turnaround(self):
        """Validate that the flight plan has no turns greater than 180 degrees.
        Be careful: sometimes it is truly necessary to have a turnaround, e.g., closed airspace,
        wrong landing sequence...
        
        Returns:
            bool: True if flight plan is valid (no turns > 180 degrees), False otherwise
        """
        if len(self.waypoints) < 3:
            return True  # Not enough waypoints to have a turn-around
            
        for i in range(1, len(self.waypoints)-1):
            # Get vectors between consecutive waypoints
            v1 = self.waypoints[i] - self.waypoints[i-1]  
            v2 = self.waypoints[i+1] - self.waypoints[i]
            
            # Calculate angle between vectors using dot product
            # Normalize vectors first to avoid numerical issues
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Convert to degrees
            angle_deg = np.degrees(angle)
            
            # Check if turn is greater than 180 degrees
            if angle_deg > 180:
                return False
                
        return True