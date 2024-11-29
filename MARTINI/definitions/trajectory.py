import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self, waypoints, altitudes, speeds, cat, t0 = 0):
        """Initialize a trajectory with waypoints, altitudes, and speeds.
        
        Args:
            waypoints (np.ndarray): Array of shape (N, 2) containing lat/lon coordinates
            altitudes (list): List of altitudes in feet for each waypoint
            speeds (list): List of speeds in km/s for each waypoint
        """
        # Convert waypoints to numpy array if not already
        self.waypoints = np.array(waypoints)
        self.altitudes = np.array(altitudes)
        self.speeds = np.array(speeds)
        self.cat = cat # for wake turbulence separation
        self.passing_time = None 
        self.t0 = t0

        self.compute_times() # Compute passing times for each waypoint
        
        # Validate inputs
        if self.waypoints.shape[1] != 2:
            raise ValueError("Waypoints must be 2D coordinates (lat, lon)")
        
        if not (len(self.waypoints) == len(self.altitudes) == len(self.speeds)):
            raise ValueError("Number of waypoints, altitudes, and speeds must match")
        

    def plot(self, ax=None, show_annotations=True):
        """Plot the trajectory on a map with optional annotations.
        
        Args:
            ax (GeoAxes, optional): Cartopy axis for plotting. If None, creates new figure
            show_annotations (bool): Whether to show altitude and speed annotations
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.stock_img()
            
        # Plot trajectory line
        ax.plot(self.waypoints[:, 1], self.waypoints[:, 0],
                'r-', transform=ccrs.PlateCarree(), linewidth=2)
        
        # Plot waypoints
        ax.scatter(self.waypoints[:, 1], self.waypoints[:, 0],
                  c='red', transform=ccrs.PlateCarree(), zorder=5)
        
        if show_annotations:
            for i, (lon, lat) in enumerate(self.waypoints):
                # Create annotation text
                annotation = f"Alt: {self.altitudes[i]:.0f} ft\nSpeed: {self.speeds[i]:.1f} km/s"
                
                # Add annotation with offset
                ax.annotate(annotation, 
                           xy=(lon, lat), 
                           xytext=(10, 10),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                           transform=ccrs.PlateCarree())
        
        return ax

    def compute_times(self):
        """Compute the time at each waypoint based on distances and speeds.
        
        Returns:
            np.ndarray: Array of times (in seconds) at each waypoint, with first waypoint at t=0
        """
        # Initialize times array
        times = np.zeros(len(self.waypoints))
        
        # For each pair of consecutive waypoints
        for i in range(len(self.waypoints)-1):
            # Convert lat/lon to radians for haversine formula
            lat1, lon1 = np.radians(self.waypoints[i])
            lat2, lon2 = np.radians(self.waypoints[i+1])
            
            # Haversine formula for great circle distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            # Earth radius in kilometers
            R = 6371
            distance = R * c
            
            # Calculate time for this segment using average speed
            avg_speed = (self.speeds[i] + self.speeds[i+1]) / 2
            segment_time = distance / avg_speed
            
            # Add to cumulative time
            times[i+1] = times[i] + segment_time

        # Set passing_time property to times
        self.passing_time = times + self.t0
        
        return times + self.t0
