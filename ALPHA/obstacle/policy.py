import numpy as np
from .obstacle import Obstacle
from typing import List
from MARTINI.airspace.geo import point_in_polygon


def get_polygon_centroid(polygon):
    """Helper function to compute polygon centroid"""
    return np.mean(polygon, axis=0)

def get_distance_between_polygons(poly1, poly2):
    """Helper function to compute approximate distance between polygons using centroids"""
    return np.linalg.norm(get_polygon_centroid(poly1) - get_polygon_centroid(poly2))

def sample_action(obstacles: Obstacle, intersecting_obstacle_indices: List[int], 
                 max_distance: float = 5.0, min_distance: float = 1.0, num_samples: int = 1, 
                 nearby_threshold: float = 3.0):
    """
    Sample actions prioritizing points near intersecting obstacles and their nearby neighbors.
    
    Args:
        obstacles: List of Obstacle objects
        intersecting_obstacle_indices: List of indices of intersecting obstacles
        max_distance: Maximum distance from obstacle boundary to sample (kilometers)
        min_distance: Minimum distance from obstacle boundary to sample (kilometers)
        num_samples: Number of points to sample
        nearby_threshold: Distance threshold to consider obstacles as "nearby" (kilometers)
    """
    # Collect all polygons and assign weights
    all_polygons = []
    polygon_weights = []
    polygon_to_obstacle_idx = []  # Keep track of which obstacle each polygon belongs to
    
    # First, collect intersecting obstacles' polygons
    intersecting_polygons = set()
    for poly in obstacles.get_polygons():
        intersecting_polygons.add(tuple(map(tuple, poly)))
    
    # Process all obstacles and their polygons
    for obs_idx, poly in enumerate(obstacles.get_polygons()):
        poly_array = np.array(poly)
        all_polygons.append(poly_array)
        polygon_to_obstacle_idx.append(obs_idx)
        
        # Assign weights based on obstacle type
        if obs_idx in intersecting_obstacle_indices:
            polygon_weights.append(1.0)  # Highest weight for intersecting obstacles
        else:
            # Check if this polygon is near any intersecting polygon
            is_nearby = False
            for intersecting_poly in intersecting_polygons:
                if get_distance_between_polygons(poly_array, np.array(intersecting_poly)) < nearby_threshold:
                    is_nearby = True
                    break
            
            polygon_weights.append(0.5 if is_nearby else 0.1)  # Higher weight for nearby obstacles
    
    if not all_polygons:
        return np.random.uniform(-max_distance, max_distance, size=(num_samples, 2))
    
    # Normalize weights
    polygon_weights = np.array(polygon_weights)
    polygon_weights /= polygon_weights.sum()
    
    # Generate samples
    oversample_factor = 3
    total_samples = num_samples * oversample_factor
    samples = []
    
    # Sample polygons based on weights
    selected_polygons = np.random.choice(
        len(all_polygons), 
        size=total_samples, 
        p=polygon_weights
    )
    
    for poly_idx in selected_polygons:
        polygon = all_polygons[poly_idx]
        num_vertices = len(polygon)
        
        # Randomly select an edge
        edge_idx = np.random.randint(0, num_vertices)
        p1 = polygon[edge_idx]
        p2 = polygon[(edge_idx + 1) % num_vertices]
        
        # Sample a point along the edge
        t = np.random.random()
        edge_point = p1 + t * (p2 - p1)
        
        # Sample distance from edge using exponential distribution
        # Use smaller scale for intersecting obstacles
        if polygon_to_obstacle_idx[poly_idx] in intersecting_obstacle_indices:
            distance = np.random.exponential(max_distance/4)  # Closer sampling for intersecting obstacles
        else:
            distance = np.random.exponential(max_distance/3)  # Regular sampling for others
        
        distance = min(distance, max_distance)

        distance = distance + min_distance
        
        # Sample random angle
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Generate point at sampled distance and angle from edge point
        offset = distance * np.array([np.cos(angle), np.sin(angle)])
        sampled_point = edge_point + offset


        # Check if the sampled point is not inside any polygon, then it can be admitted
        if not point_in_polygon(sampled_point, polygon):
            samples.append(sampled_point)
        
    
    # Convert to numpy array and return requested number of samples
    samples = np.array(samples)
    selected_indices = np.random.choice(len(samples), size=num_samples, replace=False)
    return samples[selected_indices]
    