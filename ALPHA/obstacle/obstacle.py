import numpy as np

class Obstacle:
    def __init__(self):
        # List of polygons representing obstacles
        # Each polygon is a list of (x,y) coordinates defining its vertices
        # Coordinates are in meters
        self.polygons = []
        
    def add_polygon(self, vertices):
        """
        Add a polygon obstacle defined by its vertices
        Args:
            vertices: List of (x,y) tuples defining polygon vertices in counter-clockwise order
        """
        self.polygons.append(np.array(vertices))
        
    def get_polygons(self):
        """
        Returns list of all obstacle polygons
        """
        return self.polygons
    
    def __str__(self):
        return f"Obstacle with {len(self.polygons)} polygons"
