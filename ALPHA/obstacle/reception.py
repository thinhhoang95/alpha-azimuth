# For the obstacle avoidance policy, the reception works as follows:
# 1. It will try to fly directly towards the destination
# 2. If it is too close to an obstacle, it will add the obstacle to the list
# 3. It will output the indices of the obstacles that it is too close to

import numpy as np
from .obstacle import Obstacle
from .policy import sample_action

def _line_segment_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """
    Helper function to detect if two line segments intersect.
    p1, p2 define the first line segment
    p3, p4 define the second line segment
    """
    # Calculate the direction vectors
    d1 = p2 - p1
    d2 = p4 - p3

    # Calculate the cross product denominator
    cross_prod = np.cross(d1, d2)
    
    # If lines are parallel (cross product = 0), return False
    if abs(cross_prod) < 1e-10:
        return False

    # Calculate t1 and t2 parameters
    t1 = np.cross(p3 - p1, d2) / cross_prod
    t2 = np.cross(p3 - p1, d1) / cross_prod

    # Check if intersection point lies within both line segments
    return (0 <= t1 <= 1) and (0 <= t2 <= 1)

def get_obstacle_indices(from_point: np.ndarray, to_point: np.ndarray, 
                                              obstacles: Obstacle, margin_km: float = 1.0) -> list:
    """
    Detect which obstacles intersect with the line segment from from_point to to_point.
    
    Args:
        from_point: Starting point of the line segment
        to_point: Ending point of the line segment
        obstacles: Obstacle object containing polygons
        margin_km: Safety margin in kilometers (not used in basic intersection check)
        
    Returns:
        List of indices of intersecting obstacles
    """
    intersecting_obstacles = []
    
    for obstacle_idx, polygon in enumerate(obstacles.get_polygons()):
        # Check intersection with each edge of the polygon
        n_vertices = len(polygon)
        for i in range(n_vertices):
            # Get current edge vertices
            p3 = polygon[i]
            p4 = polygon[(i + 1) % n_vertices]  # Wrap around to first vertex
            
            # Check if the line segment intersects with this edge
            if _line_segment_intersection(from_point, to_point, p3, p4):
                intersecting_obstacles.append(obstacle_idx)
                break  # Move to next obstacle once intersection is found
                
    return intersecting_obstacles

def get_attention_weight(from_point: np.ndarray, to_point: np.ndarray, obstacles: Obstacle, margin_km: float = 1.0):
    obstacle_indices = get_obstacle_indices(from_point, to_point, obstacles, margin_km)
    if len(obstacle_indices) == 0:
        return 0.0, None # No obstacle in the way, no need to use the obstacle avoidance policy
    else:
        return 1.0, obstacle_indices[0] # Obstacle in the way, need to use the obstacle avoidance policy
    
def propose_action(from_point: np.ndarray, to_point: np.ndarray, obstacles: Obstacle, margin_km: float = 1.0):
    attention_weight, obstacle_index = get_attention_weight(from_point, to_point, obstacles, margin_km)
    if attention_weight == 0.0:
        return attention_weight, None
    else:
        return attention_weight, sample_action(obstacles, [obstacle_index], max_distance=10.0, min_distance=1.0, num_samples=100, nearby_threshold=3.0)

