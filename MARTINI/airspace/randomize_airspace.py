import numpy as np

def sample_airspace_radius(loc: float, beta: float = 1.5, N: int = 1000) -> float:
    """Sample a radius from an Exponential Power Distribution.
    
    This creates a distribution of aircraft that is more concentrated near the center
    but still allows for some aircraft at larger distances.

    Args:
        loc (float): Mean radius in meters to sample around
        beta (float): Shape parameter > 1 gives lighter tails than exponential
        N (int): Number of samples to draw

    Returns:
        float: Sampled area in meters squared
    """
    scale = loc / beta  # Adjust scale to match desired mean radius
    
    # Sample N values from exponential power distribution
    r = np.zeros(N)
    for i in range(N):
        x = np.random.exponential(scale=1.0)
        u = np.random.uniform(0, 1)
        r[i] = scale * (x ** (1/beta)) if u <= np.exp(-(x**beta - x)) else sample_airspace_radius(loc, beta, 1)
    
    return r

def get_num_points_of_airspace(n_min: int = 4, n_max: int = 8, p: np.ndarray = None, N: int = 1000) -> int:
    """Sample number of points for airspace from Categorical distribution.
    
    Args:
        n_min (int): Minimum number of points (default: 4)
        n_max (int): Maximum number of points (default: 8)
        
    Returns:
        int: Sampled number of points
    """
    # Create array of possible values from n_min to n_max
    values = np.arange(n_min, n_max + 1)

    if p is None:
        # Equal probability for each value
        probs = np.ones(len(values)) / len(values)
    else:
        probs = p

    # If length of p does not match number of values, raise an error
    if len(probs) != len(values):
        raise ValueError("Length of p must match number of values")
    
    # Sample N values from categorical distribution
    return np.random.choice(values, size=N, p=probs)


from typing import Tuple

def generate_unit_polygon(n_points: int, radius: float, min_angle: float = 60, max_attempts: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Generate a random simple (non-self-intersecting) polygon with approximately unit area.
    
    Args:
        n_points (int): Number of vertices in the polygon
        max_attempts (int): Maximum number of attempts for rejection sampling
        
    Returns:
        numpy.ndarray: Array of shape (n_points, 2) containing the vertices
        float: Actual area of the generated polygon
    """
    def compute_area(points):
        """Compute area of polygon using shoelace formula"""
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if line segments (p1,p2) and (p3,p4) intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def has_self_intersections(points):
        """Check if polygon has any self-intersections"""
        n = len(points)
        # Check each pair of non-adjacent segments
        for i in range(n):
            for j in range(i + 2, n):
                # Skip if segments share an endpoint
                if (i == 0 and j == n-1): # or (j == i + 2):
                    continue
                
                if segments_intersect(points[i], points[(i+1)%n], 
                                   points[j], points[(j+1)%n]):
                    return True
        return False
    
    def compute_angle(p1, p2, p3):
        """Compute angle between vectors (p2-p1) and (p3-p2) in degrees"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Handle numerical precision issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle
    
    def has_sharp_angles(points, min_angle):
        """Check if polygon has any angles smaller than min_angle"""
        n = len(points)
        for i in range(n):
            prev_point = points[i-1]  # Previous point (wraps around)
            curr_point = points[i]
            next_point = points[(i+1)%n]  # Next point (wraps around)
            
            angle = compute_angle(prev_point, curr_point, next_point)
            if angle < min_angle or angle > 360 - min_angle:
                return True
        return False
    
    def generate_candidate():
        # Generate points on unit circle with random radii between radius/2 and radius
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
        radii = np.random.uniform(radius/2, 2 * radius, n_points)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y])
    
    best_points = None
    best_area_diff = float('inf')
    
    for attempt in range(max_attempts):
        points = generate_candidate()
        
        # Skip self-intersecting polygons
        if has_self_intersections(points) or has_sharp_angles(points, min_angle):
            continue
            
        area = compute_area(points)
        area_diff = abs(area - radius**2)
        
        if area_diff < best_area_diff:
            best_points = points
            best_area_diff = area_diff
            
        # If we're close enough, return early
        if area_diff < 10.0:
            return best_points, area
    
    # If we couldn't get very close, return the best we found
    # (or None if all attempts produced self-intersecting polygons)
    if best_points is None:
        raise ValueError("Could not generate a valid non-self-intersecting polygon")
        
    return best_points, compute_area(best_points)