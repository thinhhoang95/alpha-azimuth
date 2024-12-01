import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

EARTH_RADIUS = 6371 # km

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
        self.waypoints_xyz = None # Cartesian coordinates
        self.altitudes = np.array(altitudes)
        self.speeds = np.array(speeds)
        self.cat = cat # for wake turbulence separation
        self.passing_time = None 
        self.t0 = t0

        self.compute_times() # Compute passing times for each waypoint
        self.compute_cartesian_coordinates() # Compute Cartesian coordinates for each waypoint
        
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
            for i, (lat, lon) in enumerate(self.waypoints):
                annotation = f"{i}: FL{self.altitudes[i]/100:.0f} / {(self.speeds[i] * 3600 / 1.852):.0f}\nT={self.passing_time[i]:.0f}"
                ax.text(lon, lat, annotation,
                       transform=ccrs.PlateCarree(),
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.3),
                       fontsize=8)  # Added small font size
        
        return ax

    def compute_times(self, t0 = None):
        """Compute the time at each waypoint based on distances and speeds.
        
        Returns:
            np.ndarray: Array of times (in seconds) at each waypoint, with first waypoint at t=0
        """
        if t0 is not None:
            self.t0 = t0
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
            distance = R * c # in km
            
            # Calculate time for this segment using average speed
            avg_speed = (self.speeds[i] + self.speeds[i+1]) / 2
            segment_time = distance / avg_speed # in seconds
            
            # Add to cumulative time
            times[i+1] = times[i] + segment_time

        # Set passing_time property to times
        self.passing_time = times + self.t0
        
        return times + self.t0
    
    def compute_cartesian_coordinates(self):
        """Convert waypoints from lat/lon to Cartesian (x, y, z) coordinates on a unit sphere.
        
        Returns:
            np.ndarray: Array of Cartesian coordinates for each waypoint
        """
        # Convert degrees to radians
        lat_rad = np.deg2rad(self.waypoints[:, 0])
        lon_rad = np.deg2rad(self.waypoints[:, 1])
        
        # Convert to Cartesian coordinates on a unit sphere
        x = np.cos(lat_rad) * np.cos(lon_rad) * EARTH_RADIUS
        y = np.cos(lat_rad) * np.sin(lon_rad) * EARTH_RADIUS
        z = np.sin(lat_rad) * EARTH_RADIUS
        
        # Store Cartesian coordinates 
        self.waypoints_xyz = np.column_stack((x, y, z))
        
        return self.waypoints_xyz

    def __str__(self):
        output = []
        for i, (waypoint, speed, altitude, time) in enumerate(zip(self.waypoints, self.speeds, self.altitudes, self.passing_time)):
            output.append(f"Waypoint {i}: lat={waypoint[0]:.4f}, lon={waypoint[1]:.4f}, speed={speed:.1f} km/s, altitude={altitude:.1f} ft, time={time:.1f} s")
        return "\n".join(output)