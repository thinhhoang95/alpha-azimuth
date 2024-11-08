import math

def subtract(p1, p2):
    """Subtract two points/vector: p1 - p2."""
    return (p1[0] - p2[0], p1[1] - p2[1])

def dot(v1, v2):
    """Dot product of two vectors."""
    return v1[0]*v2[0] + v1[1]*v2[1]

def cross(v1, v2):
    """Cross product of two vectors."""
    return v1[0]*v2[1] - v1[1]*v2[0]

def line_intersection(p, r, q, s):
    """
    Find the intersection point of two lines:
    Line 1: p + t*r
    Line 2: q + u*s
    Returns (t, u) if they intersect, else None.
    """
    r_cross_s = cross(r, s)
    q_p = subtract(q, p)
    q_p_cross_r = cross(q_p, r)
    
    if r_cross_s == 0:
        if q_p_cross_r == 0:
            # Lines are colinear
            return None
        else:
            # Lines are parallel and non-intersecting
            return None
    t = cross(q_p, s) / r_cross_s
    u = cross(q_p, r) / r_cross_s
    return (t, u)

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    :param point: Tuple (x, y)
    :param polygon: List of tuples [(x1, y1), (x2, y2), ...]
    :return: True if point is inside polygon, False otherwise
    """
    x, y = point
    inside = False
    
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
            x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
            
    return inside

def find_intersection(polygon, velocity, start_point):
    """
    Find the intersection point of a ray starting at start_point in the direction of velocity
    with the polygon's edges.
    
    :param polygon: List of tuples [(x1, y1), (x2, y2), ...]
    :param velocity: Tuple (vx, vy) as a unit vector
    :param start_point: Tuple (x, y) on the border of the polygon
    :return: Tuple (intersection_point, valid_path) where intersection_point is (x, y) or None,
            and valid_path is True if the path stays inside the polygon
    """
    intersections = []
    num_vertices = len(polygon)
    p = start_point
    r = velocity

    for i in range(num_vertices):
        q = polygon[i]
        s = subtract(polygon[(i + 1) % num_vertices], q)

        result = line_intersection(p, r, q, s)
        if result is not None:
            t, u = result
            if t > 1e-9 and 0 <= u <= 1:
                intersection_point = (p[0] + t * r[0], p[1] + t * r[1])
                intersections.append((t, intersection_point))

    if not intersections:
        return None, False

    # Sort and get closest intersection
    intersections.sort(key=lambda x: x[0])
    t, intersection_point = intersections[0]
    
    # Check if path stays inside polygon
    num_checks = 10  # Number of points to check along the path
    valid_path = True
    
    for i in range(1, num_checks):
        # Check points along the path
        check_t = (t * i) / num_checks
        check_point = (p[0] + check_t * r[0], p[1] + check_t * r[1])
        if not point_in_polygon(check_point, polygon):
            valid_path = False
            break
    
    return intersection_point, valid_path

# Example Usage
if __name__ == "__main__":
    # Define a non-convex polygon (e.g., a star shape)
    polygon = [
        (0, 0),
        (2, 4),
        (4, 0),
        (3, 3),
        (5, 5),
        (2, 3),
        (0, 5),
        (1, 3),
        (-2, 3),
        (0, 0)
    ]

    # Ensure the polygon is closed by removing duplicate last point if necessary
    if polygon[0] == polygon[-1]:
        polygon = polygon[:-1]

    # Define a starting point on the border
    start_point = (2, 4)  # This is one of the vertices

    # Define a velocity vector (unit vector)
    angle_degrees = 45
    angle_radians = math.radians(angle_degrees)
    velocity = (math.cos(angle_radians), math.sin(angle_radians))

    # Find the intersection
    intersection, valid_path = find_intersection(polygon, velocity, start_point)

    print(f"Intersection Point: {intersection}")
    print(f"Valid Path: {valid_path}")
