from typing import List, Tuple
import numpy as np

def polygon_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the signed area of a polygon to determine its orientation.

    Args:
        points: List of 2D points (x, y) defining the polygon vertices.

    Returns:
        float: Signed area of the polygon.
               Positive if counter-clockwise, negative if clockwise.
    """
    area = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        area += (x0 * y1 - x1 * y0)
    return area / 2.0

def find_point_index(points: List[Tuple[float, float]], point: Tuple[float, float]) -> int:
    """
    Find the index of a point in the points list.

    Args:
        points: List of 2D points (x, y).
        point: The point (x, y) to find.

    Returns:
        int: Index of the point in the list.

    Raises:
        ValueError: If the point is not found in the list.
    """
    try:
        return points.index(point)
    except ValueError:
        raise ValueError("Point not found in the points list.")

def get_inward_normal(points: List[Tuple[float, float]], edge: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the inward-pointing normal unit vector for a polygon edge.

    Args:
        points: List of 2D points (x, y) defining the polygon vertices.
        edge: List containing two points (p1, p2) that define the edge.

    Returns:
        Tuple[float, float]: Inward normal unit vector (nx, ny).
    """
    if len(edge) != 2:
        raise ValueError("Edge must be a list of two points.")

    # Determine polygon orientation
    area = polygon_area(points)
    ccw = area > 0

    # Extract edge points
    p1, p2 = edge

    # Calculate edge vector
    edge_x = p2[0] - p1[0]
    edge_y = p2[1] - p1[1]

    if ccw:
        # For CCW, inward normal is rotated 90° clockwise
        normal_x = edge_y
        normal_y = -edge_x
    else:
        # For CW, inward normal is rotated 90° counter-clockwise
        normal_x = -edge_y
        normal_y = edge_x

    # Normalize the normal vector
    length = (normal_x**2 + normal_y**2)**0.5
    if length == 0:
        raise ValueError("Edge length is zero; cannot compute normal.")
    normal_x /= length
    normal_y /= length

    return (normal_x, normal_y)

def find_line_polygon_intersection(point, velocity, polygon):
    """
    Find intersection of a ray with a polygon.
    
    Args:
        point (np.array): Starting point [x, y]
        velocity (np.array): Direction vector [dx, dy]
        polygon (np.array): Nx2 array of polygon vertices
    
    Returns:
        np.array: Intersection point [x, y] or None if no intersection
    """
    # Create a distant point along the velocity vector
    # (making sure it's well beyond the polygon)
    far_point = point + velocity * 10000  # Large enough to cross polygon
    
    closest_intersection = None
    min_distance = float('inf')
    
    # Check intersection with each polygon edge
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        
        # Line segment intersection calculation
        x1, y1 = point
        x2, y2 = far_point
        x3, y3 = p1
        x4, y4 = p2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:  # Lines are parallel
            continue
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = np.array([
                x1 + t * (x2 - x1),
                y1 + t * (y2 - y1)
            ])
            
            # Keep only the closest intersection point
            dist = np.linalg.norm(intersection - point)
            if dist < min_distance:
                min_distance = dist
                closest_intersection = intersection

    # Check if closest intersection is too close to starting point
    if closest_intersection is not None:
        if np.linalg.norm(closest_intersection - point) < 1.0:
            return None
    
    return closest_intersection



def compute_transit_length(points: List[Tuple[float, float]], entry_point: Tuple[float, float], velocity: Tuple[float, float]) -> float:
    """
    Compute the length of path from entry point to exit point through a polygon, given a velocity vector.
    
    Args:
        points: List of 2D points (x, y) defining the polygon vertices.
        entry_point: Point (x, y) on polygon boundary where path enters.
        velocity: Unit velocity vector (vx, vy) defining direction of travel.
        
    Returns:
        float: Length of path from entry to exit point.
        
    Raises:
        ValueError: If entry point is not on polygon boundary or velocity is not a unit vector.
    """
    # Validate velocity is unit vector
    vel_mag = (velocity[0]**2 + velocity[1]**2)**0.5
    if abs(vel_mag - 1.0) > 1e-10:
        raise ValueError("Velocity must be a unit vector")
        
    # Create line extending through polygon
    large_dist = 1e6  # Large enough to extend beyond polygon
    line_end = (
        entry_point[0] + large_dist * velocity[0],
        entry_point[1] + large_dist * velocity[1]
    )
    line_start = (
        entry_point[0] - large_dist * velocity[0], 
        entry_point[1] - large_dist * velocity[1]
    )
    
    # Find all intersection points with polygon edges
    intersections = []
    n = len(points)
    for i in range(n):
        edge_start = points[i]
        edge_end = points[(i + 1) % n]
        
        # Check if lines intersect
        denom = (
            (edge_end[1] - edge_start[1]) * (line_end[0] - line_start[0]) -
            (edge_end[0] - edge_start[0]) * (line_end[1] - line_start[1])
        )
        
        if abs(denom) < 1e-10:  # Lines are parallel
            continue
            
        t = (
            (edge_end[0] - edge_start[0]) * (line_start[1] - edge_start[1]) -
            (edge_end[1] - edge_start[1]) * (line_start[0] - edge_start[0])
        ) / denom
        
        s = (
            (line_end[0] - line_start[0]) * (line_start[1] - edge_start[1]) -
            (line_end[1] - line_start[1]) * (line_start[0] - edge_start[0])
        ) / denom
        
        if 0 <= s <= 1 and 0 <= t <= 1:  # Lines intersect within segments
            x = line_start[0] + t * (line_end[0] - line_start[0])
            y = line_start[1] + t * (line_end[1] - line_start[1])
            intersections.append((x, y))
    
    if len(intersections) < 2:
        raise ValueError("Path does not properly intersect polygon boundary twice")
        
    # Find the two points closest to entry point
    distances = [(p[0] - entry_point[0])**2 + (p[1] - entry_point[1])**2 for p in intersections]
    closest_idx = distances.index(min(distances))
    
    # Get the other intersection point
    if closest_idx == 0:
        exit_point = intersections[1]
    else:
        exit_point = intersections[0]
        
    # Compute transit length
    return ((exit_point[0] - entry_point[0])**2 + (exit_point[1] - entry_point[1])**2)**0.5


def get_track_angle(point_from: Tuple[float, float], point_to: Tuple[float, float]) -> float:
    """
    Calculate the track angle between two points in degrees.
    
    Args:
        point_from: Starting point (x, y)
        point_to: Ending point (x, y)
        
    Returns:
        float: Track angle in degrees (0-360), measured clockwise from true north
    """
    dx = point_to[0] - point_from[0]
    dy = point_to[1] - point_from[1]
    
    # Calculate angle in radians from east axis
    angle = np.arctan2(dy, dx)
    
    # Convert to degrees from north axis
    degrees = 90 - np.degrees(angle)
    
    # Normalize to 0-360 range
    if degrees < 0:
        degrees += 360
        
    return degrees




def compute_polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the centroid (geometric center) of a polygon.
    
    Args:
        points: List of 2D points (x, y) defining the polygon vertices
        
    Returns:
        Tuple[float, float]: Centroid coordinates (x, y)
        
    Raises:
        ValueError: If fewer than 3 points provided
    """
    if len(points) < 3:
        raise ValueError("At least 3 points required to compute polygon centroid")
        
    # Initialize area accumulator and centroid coordinates
    area = 0
    cx = 0
    cy = 0
    
    # Process each vertex pair
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        # Get current and next vertex
        xi, yi = points[i]
        xj, yj = points[j]
        
        # Compute signed area contribution
        cross = xi * yj - xj * yi
        area += cross
        
        # Accumulate centroid coordinates
        cx += (xi + xj) * cross
        cy += (yi + yj) * cross
    
    # Complete the area calculation
    area = area / 2.0
    
    # Guard against division by zero
    if area == 0:
        raise ValueError("Polygon has zero area")
        
    # Complete centroid calculation
    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    
    return (cx, cy)


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Determine if a point lies inside a polygon using ray casting algorithm.
    
    Args:
        point: Point to test (x, y)
        polygon: List of 2D points (x, y) defining the polygon vertices
        
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    inside = False
    
    # Cast ray from point to the right
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        
        yi, yj = polygon[i][1], polygon[j][1]
        xi, xj = polygon[i][0], polygon[j][0]
        
        # Check if point is exactly on boundary
        if (x, y) == (xi, yi) or (x, y) == (xj, yj):
            return True
            
        # Check if line segment intersects with ray
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
            
    return inside





def line_segment_intersects_polygon(
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    polygon: List[Tuple[float, float]],
    buffer: float = 0.0
) -> bool:
    """
    Check if a line segment intersects with or touches a polygon.

    Args:
        line_start: Starting point of line segment (x, y)
        line_end: Ending point of line segment (x, y)
        polygon: List of 2D points (x, y) defining the polygon vertices
        buffer: Optional safety buffer distance around polygon (default: 0.0)

    Returns:
        bool: True if line segment intersects/touches polygon, False otherwise
    """
    # First check if either endpoint is inside the polygon
    if buffer == 0.0:
        if point_in_polygon(line_start, polygon) or point_in_polygon(line_end, polygon):
            return True

    # Check intersection with each polygon edge
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        
        # If using buffer, offset the polygon edge
        if buffer > 0.0:
            # Get normal vector for the edge
            normal = get_inward_normal(polygon, [p1, p2])
            # Create buffered points
            p1_buffered = (p1[0] + buffer * normal[0], p1[1] + buffer * normal[1])
            p2_buffered = (p2[0] + buffer * normal[0], p2[1] + buffer * normal[1])
            p1, p2 = p1_buffered, p2_buffered

        # Line segment intersection calculation
        x1, y1 = line_start
        x2, y2 = line_end
        x3, y3 = p1
        x4, y4 = p2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:  # Lines are parallel
            continue
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        # Check if intersection occurs within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True

    return False



