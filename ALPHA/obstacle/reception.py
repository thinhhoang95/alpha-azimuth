# For the obstacle avoidance policy, the reception works as follows:
# 1. It will try to fly directly towards the destination
# 2. If it is too close to an obstacle, it will add the obstacle to the list
# 3. It will output the indices of the obstacles that it is too close to

import numpy as np
from .obstacle import Obstacle

def get_obstacle_indices_using_ray_intersection(from_point: np.ndarray, to_point: np.ndarray, obstacles: Obstacle, margin_km: float = 1.0):
    """
    Returns indices of obstacles that intersect with the ray from from_point to to_point.
    Attention: it checks for ray intersection, not segment intersection.
    
    Args:
        from_point: Starting point as np.array([x, y])
        to_point: End point as np.array([x, y])
        obstacles: Obstacle object containing polygons
        margin_km: Safety margin in kilometers to add around obstacles
        
    Returns:
        List of indices of intersecting obstacles
    """
    def ray_intersects_segment(p1, p2): # p1, p2 are two vertices of a polygon
        # Convert margin to meters
        margin = margin_km * 1000
        
        # Expand polygon points by margin
        p1 = p1 + np.sign(p1 - from_point) * margin
        p2 = p2 + np.sign(p2 - from_point) * margin
        
        # Ray direction
        ray_dir = to_point - from_point
        segment_dir = p2 - p1
        
        # Cross products
        denom = np.cross(ray_dir, segment_dir)
        if abs(denom) < 1e-10:  # Lines are parallel
            return False, None
            
        t = np.cross(p1 - from_point, segment_dir) / denom
        u = np.cross(p1 - from_point, ray_dir) / denom
        
        if (t >= 0) and (0 <= u <= 1):
            intersection_point = from_point + t * ray_dir
            return True, intersection_point
        return False, None

    # List to store (index, distance) pairs
    intersections = []
    
    for i, polygon in enumerate(obstacles.get_polygons()):
        n_vertices = len(polygon)
        for j in range(n_vertices):
            p1 = polygon[j]
            p2 = polygon[(j + 1) % n_vertices]
            
            intersects, intersection_point = ray_intersects_segment(p1, p2)
            if intersects:
                # Calculate distance from start point to intersection
                distance = np.linalg.norm(intersection_point - from_point)
                intersections.append((i, distance))
                break
    
    # Sort by distance and extract indices
    intersecting_indices = [idx for idx, dist in sorted(intersections, key=lambda x: x[1])]
    return intersecting_indices
    