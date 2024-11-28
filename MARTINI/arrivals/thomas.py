import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

import numpy as np
from shapely.geometry import Polygon, LineString, Point

from MARTINI.airspace import geo

def thomas_point_process_border(points: List[Tuple[float, float]], lambda_parent: float, mu: float, sigma: float, random_state: int = None, min_dist_to_vertex: float = 50, min_transit_length: float = 50) -> Tuple[List[Tuple[float, float]], List[int], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    Generate random points along the border of a polygon using the Thomas Point Process.
    ATTENTION: do not use this function. Use Thomas-Jensen process instead.

    Parameters:
    - points: List of (x, y) tuples defining the polygon vertices in order.
    - lambda_parent: Intensity of parent points per unit length (λ).
    - mu: Mean number of offspring per parent (μ).
    - sigma: Standard deviation for the Gaussian displacement of offspring from parents (σ).
    - random_state: (Optional) Seed or NumPy RandomState for reproducibility.

    Returns:
    - Tuple containing:
        - offspring_points: List of (x, y) tuples representing the generated points
        - parent_indices: List of integers indicating the parent index for each offspring
        - edge_points: List of ((x1,y1), (x2,y2)) tuples representing the endpoints of the edge each offspring belongs to
    """
    # Initialize the random number generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    
    # Create a Shapely Polygon from the given points
    polygon = Polygon(points)
    if not polygon.is_valid:
        raise ValueError("Invalid polygon provided. Please ensure the polygon is well-defined.")
    
    # Extract the exterior boundary as a LineString
    boundary = polygon.exterior
    total_length = boundary.length

    # Step 1: Generate the number of parent points based on Poisson distribution
    expected_parents = lambda_parent * total_length
    num_parents = rng.poisson(expected_parents)

    # Calculate cumulative lengths of edges
    edge_lengths = [LineString([points[i], points[i+1]]).length 
                   for i in range(len(points)-1)]
    cumulative_lengths = np.cumsum([0] + edge_lengths)
    
    # Step 2: Generate parent positions with bias towards shorter edges
    edge_weights = 1 / np.array(edge_lengths)  # Inverse relationship
    edge_weights = edge_weights / edge_weights.sum()  # Normalize to probabilities
    
    # Assign number of parents to each edge based on weights
    edge_parents = rng.multinomial(num_parents, edge_weights)
    
    # Generate parent positions within each edge
    parent_positions = []
    for i, n_parents in enumerate(edge_parents):
        if n_parents > 0:
            # Get the start position for this edge
            edge_start = cumulative_lengths[i]
            edge_length = edge_lengths[i]
            # Generate uniform positions within this edge
            edge_positions = edge_start + rng.uniform(0, edge_length, n_parents)
            parent_positions.extend(edge_positions.tolist())
    
    parent_positions = np.array(parent_positions)
    
    # Step 3: For each parent, determine the number of offspring
    num_offspring_per_parent = rng.poisson(mu, num_parents)
    
    # Total number of offspring points
    total_offspring = num_offspring_per_parent.sum()
    if total_offspring == 0:
        return [], [], []
    
    # Create parent indices for each offspring
    parent_indices = np.repeat(np.arange(num_parents), num_offspring_per_parent)
    
    # Generate offspring positions
    parent_expanded = np.repeat(parent_positions, num_offspring_per_parent)
    displacements = rng.normal(0, sigma, total_offspring)
    offspring_positions = (parent_expanded + displacements) % total_length
    
    # Get coordinates of polygon vertices
    coords = list(boundary.coords)
    
    # Initialize lists for offspring points and their edge endpoints
    offspring_points = []
    edge_points = []
    
    # Process each offspring position
    for pos in offspring_positions:
        # Find which edge this position belongs to
        edge_idx = np.searchsorted(cumulative_lengths, pos) - 1
        
        # Get the edge endpoints
        start_point = coords[edge_idx]
        end_point = coords[edge_idx + 1]
        
        # Store the offspring point and its edge endpoints
        point = boundary.interpolate(pos).coords[0]
        offspring_points.append(point)
        edge_points.append((start_point, end_point))

    # Final filtering.
    # Criteria 1: No offspring points should locate too close to the polygon points.
    filtered_offspring = []
    filtered_parent_indices = []
    filtered_edge_points = []

    parent_indices = parent_indices.tolist()
    
    for i, (point, parent_idx, edge) in enumerate(zip(offspring_points, parent_indices, edge_points)):
        # Check distance to all polygon vertices
        too_close = False
        for vertex in coords[:-1]:  # Exclude last point since it duplicates first
            dist = ((point[0] - vertex[0])**2 + (point[1] - vertex[1])**2)**0.5
            if dist < min_dist_to_vertex:
                too_close = True
                break
                
        if not too_close:
            filtered_offspring.append(point)
            filtered_parent_indices.append(parent_idx)
            filtered_edge_points.append(edge)
    
    # Update the lists
    offspring_points = filtered_offspring
    parent_indices = filtered_parent_indices
    edge_points = filtered_edge_points

    # Criteria 2: No offspring points should locate where the transit time is too short.
    # Filter offspring points based on minimum transit time
    filtered_offspring = []
    filtered_parent_indices = []
    filtered_edge_points = []
    filtered_transit_length = []

    for i, (point, parent_idx, edge) in enumerate(zip(offspring_points, parent_indices, edge_points)):
        # Get inward normal at the entry point as velocity direction
        velocity = geo.get_inward_normal(coords[:-1], edge)
        
        try:
            # Compute transit length through polygon
            transit_length = geo.compute_transit_length(coords[:-1], point, velocity)
            
            # Keep point if transit length is sufficient
            if transit_length >= min_transit_length:
                filtered_offspring.append(point)
                filtered_parent_indices.append(parent_idx)
                filtered_edge_points.append(edge)
                filtered_transit_length.append(transit_length)
        except ValueError:
            # Skip points that raise ValueError (invalid entry/exit)
            continue
    
    # Update lists with filtered results
    offspring_points = filtered_offspring
    parent_indices = filtered_parent_indices 
    edge_points = filtered_edge_points
    transit_lengths = filtered_transit_length

    
    return offspring_points, parent_indices, edge_points, transit_lengths








# Note that if lambda_parent is autoregressive, then we have Hawkes process.
# Thomas-Jensen-Hawkes (TJH) process.
def thomas_jensen_process(points: List[Tuple[float, float]], lambda_parent: float, mu: float, sigma: float, random_state: int = None, min_dist_to_vertex: float = 50, min_transit_length: float = 50) -> Tuple[List[Tuple[float, float]], List[int], List[Tuple[Tuple[float, float], Tuple[float, float]]], List[float]]:
    """
    Generate random points along the border of a polygon in line with the general guidelines for airspace design.
    The general criteria are to ensure the entry points are located near the middle of the edges, sufficiently far from the vertices, and the transit time is sufficient.

    Parameters:
    - points: List of (x, y) tuples defining the polygon vertices in order.
    - lambda_parent: Intensity of parent points per unit length (λ).
    - mu: Mean number of offspring per parent (μ).
    - sigma: Standard deviation for the Gaussian displacement of offspring from parents (σ).
    - random_state: (Optional) Seed or NumPy RandomState for reproducibility.
    - min_dist_to_vertex: Minimum distance (in meters) that points must be from polygon vertices.
    - min_transit_length: Minimum transit length (in meters) required through the polygon.

    Returns:
    - Tuple containing:
        - offspring_points: List of (x, y) tuples representing the generated points
        - parent_indices: List of integers indicating the parent index for each offspring
        - edge_points: List of ((x1,y1), (x2,y2)) tuples representing the endpoints of the edge each offspring belongs to
        - transit_lengths: List of float values representing the transit length through the polygon for each point
    """
    # Initialize the random number generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    
    # Create a Shapely Polygon from the given points
    polygon = Polygon(points)
    if not polygon.is_valid:
        raise ValueError("Invalid polygon provided. Please ensure the polygon is well-defined.")
    
    # Extract the exterior boundary as a LineString
    boundary = polygon.exterior
    total_length = boundary.length

    # Generate the entry distribution based on the length of the edges
    # Calculate cumulative lengths of edges
    edge_lengths = [LineString([points[i], points[i+1]]).length 
                   for i in range(len(points)-1)]
    cumulative_lengths = np.cumsum([0] + edge_lengths)
    
    # Generate parent positions with bias towards shorter edges
    # edge_weights = np.exp(-np.array(edge_lengths))  # Longer edges have less weight
    edge_weights = 1/np.array(edge_lengths)
    edge_weights = edge_weights / edge_weights.sum()  # Normalize to probabilities

    # Step 1: Generate the number of parent points based on Poisson distribution
    expected_parents = lambda_parent * total_length
    num_parents = rng.poisson(expected_parents) # total number of parents
    
    # Assign number of parents to each edge based on weights
    edge_parents = rng.multinomial(num_parents, edge_weights) # number of parents per edge, inversely proportional to edge length
    
    # Generate parent positions within each edge
    parent_positions = []
    for i, n_parents in enumerate(edge_parents):
        if n_parents > 0:
            # Get the start position for this edge
            edge_start = cumulative_lengths[i]
            edge_length = edge_lengths[i]
            # Generate uniform positions within this edge
            edge_positions = edge_start + rng.uniform(0, edge_length, n_parents)
            edge_positions = edge_start + rng.normal(loc=edge_length/2, scale=edge_length/5, size=n_parents)
            parent_positions.extend(edge_positions.tolist())
    
    parent_positions = np.array(parent_positions)
    
    # Step 3: For each parent, determine the number of offspring
    num_offspring_per_parent = rng.poisson(mu, num_parents)
    
    # Total number of offspring points
    total_offspring = num_offspring_per_parent.sum()
    if total_offspring == 0:
        return [], [], []
    
    # Create parent indices for each offspring
    parent_indices = np.repeat(np.arange(num_parents), num_offspring_per_parent)
    
    # Generate offspring positions
    parent_expanded = np.repeat(parent_positions, num_offspring_per_parent)
    displacements = rng.normal(0, sigma, total_offspring)
    offspring_positions = (parent_expanded + displacements) % total_length
    
    # Get coordinates of polygon vertices
    coords = list(boundary.coords)
    
    # Initialize lists for offspring points and their edge endpoints
    offspring_points = []
    edge_points = []
    
    # Process each offspring position
    for pos in offspring_positions:
        # Find which edge this position belongs to
        edge_idx = np.searchsorted(cumulative_lengths, pos) - 1
        
        # Get the edge endpoints
        start_point = coords[edge_idx]
        end_point = coords[edge_idx + 1]
        
        # Store the offspring point and its edge endpoints
        point = boundary.interpolate(pos).coords[0]
        offspring_points.append(point)
        edge_points.append((start_point, end_point))

    # Final filtering.
    # Criteria 1: No offspring points should locate too close to the polygon points.
    filtered_offspring = []
    filtered_parent_indices = []
    filtered_edge_points = []

    parent_indices = parent_indices.tolist()
    
    for i, (point, parent_idx, edge) in enumerate(zip(offspring_points, parent_indices, edge_points)):
        # Check distance to all polygon vertices
        too_close = False
        for vertex in coords[:-1]:  # Exclude last point since it duplicates first
            dist = ((point[0] - vertex[0])**2 + (point[1] - vertex[1])**2)**0.5
            if dist < min_dist_to_vertex:
                too_close = True
                break
                
        if not too_close:
            filtered_offspring.append(point)
            filtered_parent_indices.append(parent_idx)
            filtered_edge_points.append(edge)
    
    # Update the lists
    offspring_points = filtered_offspring
    parent_indices = filtered_parent_indices
    edge_points = filtered_edge_points

    # Criteria 2: No offspring points should locate where the transit time is too short.
    # Filter offspring points based on minimum transit time
    filtered_offspring = []
    filtered_parent_indices = []
    filtered_edge_points = []
    filtered_transit_length = []

    for i, (point, parent_idx, edge) in enumerate(zip(offspring_points, parent_indices, edge_points)):
        # Get inward normal at the entry point as velocity direction
        velocity = geo.get_inward_normal(coords[:-1], edge)
        
        try:
            # Compute transit length through polygon
            transit_length = geo.compute_transit_length(coords[:-1], point, velocity)
            
            # Keep point if transit length is sufficient
            if transit_length >= min_transit_length:
                filtered_offspring.append(point)
                filtered_parent_indices.append(parent_idx)
                filtered_edge_points.append(edge)
                filtered_transit_length.append(transit_length)
        except ValueError:
            # Skip points that raise ValueError (invalid entry/exit)
            continue
    
    # Update lists with filtered results
    offspring_points = filtered_offspring
    parent_indices = filtered_parent_indices 
    edge_points = filtered_edge_points
    transit_lengths = filtered_transit_length

    
    return offspring_points, parent_indices, edge_points, transit_lengths





from ..airspace import intersect
from ..airspace import geo
from ..airspace import randomize_airspace

def generate_problem_thomas_jensen(polygon: Tuple[np.ndarray, float], lambda_parent: float,
                                   mu: float, sigma: float, min_dist_to_vertex: float = 50,
                                   min_transit_length: float = 50, **kwargs) -> Tuple[List[Tuple[float, float]], List[int],
                                                                                      List[Tuple[Tuple[float, float],
                                                                                                 Tuple[float, float]]],
                                                                                                 List[float]]:
    """
    Generate entry and exit points for aircraft trajectories using a Thomas-Jensen point process.
    
    This function generates entry points along the polygon boundary using a Thomas-Jensen process,
    then calculates corresponding exit points and velocity vectors for valid trajectories through
    the airspace.

    The difference with Thomas-Jensen process is that this function also checks for intersections
    with the polygon boundary, and will additionally remove trajectories that do not follow the
    general airspace design criteria.

    Parameters:
        polygon (Tuple[np.ndarray, float]): Vertices of the polygon defining the airspace boundary
        lambda_parent (float): Intensity parameter for parent points in the Thomas process
        mu (float): Mean number of offspring per parent point
        sigma (float): Standard deviation for offspring displacement from parent
        min_dist_to_vertex (float, optional): Minimum allowed distance from entry points to polygon vertices. Defaults to 50.
        min_transit_length (float, optional): Minimum required transit length through the polygon. Defaults to 50.
        **kwargs: Additional keyword arguments
            - restrict_exit_to_opposite_edge (bool): If True, exit points must be on opposite edges. Defaults to False.

    Returns:
        Tuple containing:
            - List[Tuple[float, float]]: Entry points for valid trajectories
            - List[Tuple[float, float]]: Exit points for valid trajectories
            - List[np.ndarray]: Normalized velocity vectors for each trajectory

    Note:
        The function filters out invalid trajectories that don't meet minimum transit length
        requirements or have invalid intersections with the polygon boundary.
    """
    # Unpack kwargs
    restrict_exit_to_opposite_edge = kwargs.get('restrict_exit_to_opposite_edge', False)
    
    trajectory_count = 0

    # Generate points
    random_points, parent_indices, edge_points, transit_length = thomas_jensen_process(polygon, lambda_parent=lambda_parent, mu=mu, sigma=sigma, min_transit_length=min_transit_length, min_dist_to_vertex=min_dist_to_vertex, **kwargs)
    # Find the exit points for each parent
    exit_points, velocity_vectors = get_exit_point_for_parent(polygon, random_points, parent_indices, restrict_exit_to_opposite_edge=restrict_exit_to_opposite_edge)

    # If there are no points, skip
    if len(random_points) == 0:
        print("No points generated")

    # To store the final results
    final_entry_points = []
    final_exit_points = []
    final_velocity_vectors = []
    final_parent_indices = []

    # Plot the entry points
    for i, point in enumerate(random_points):
        edge = edge_points[i]

        # Get parent index for this entry point
        parent_idx = parent_indices[i]
        parent_indices_unique = np.unique(np.array(parent_indices))
        parent_idx = np.where(parent_indices_unique == parent_idx)[0][0]
        # Get corresponding velocity vector
        velocity = velocity_vectors[parent_idx]

        intersection, valid_path = intersect.find_intersection(polygon, velocity, point)
        if intersection is not None and valid_path:
            # Add check for minimum distance to vertices
            too_close_to_vertex = False
            for vertex in polygon:
                dist = np.sqrt((intersection[0] - vertex[0])**2 + (intersection[1] - vertex[1])**2)
                if dist < min_dist_to_vertex:
                    too_close_to_vertex = True
                    break
            
            if not too_close_to_vertex:
                final_entry_points.append(point)
                final_exit_points.append(intersection)
                final_velocity_vectors.append(velocity)
                final_parent_indices.append(parent_idx)
                trajectory_count += 1
        else:
            pass 
        
    return final_entry_points, final_exit_points, final_velocity_vectors, final_parent_indices











def get_exit_point_for_parent(polygon_points: List[Tuple[float, float]],
                              offspring_points: List[Tuple[float, float]], 
                              parent_indices: List[int], 
                              **kwargs) -> List[Tuple[float, float]]:
    """
    For each parent index, select one offspring point and determine the opposite edge of the polygon.
    
    Parameters:
    - polygon_points: List of (x, y) tuples defining the polygon vertices
    - offspring_points: List of (x, y) tuples representing entry points
    - parent_indices: List of integers indicating the parent index for each offspring
    
    Returns:
    - List of (x, y) tuples representing exit points on opposite edges
    """

    # Unpack kwargs
    # If restrict_exit_to_opposite_edge is True, then the exit point must be on the opposite edge.
    restrict_exit_to_opposite_edge = kwargs.get('restrict_exit_to_opposite_edge', False)


    # Create polygon for geometric operations
    polygon = Polygon(polygon_points)
    
    # Group offspring points by parent index
    parent_to_offspring = {}
    for point, parent_idx in zip(offspring_points, parent_indices):
        if parent_idx not in parent_to_offspring:
            parent_to_offspring[parent_idx] = []
        parent_to_offspring[parent_idx].append(point)
    
    exit_points = []

    velocity_vectors = []
    
    # For each parent, process one offspring point
    for parent_idx in parent_to_offspring:
        # Select first offspring point for this parent
        entry_point = parent_to_offspring[parent_idx][0]
        entry_point = Point(entry_point)
        
        # Find the closest edge to the entry point (this is the entry edge)
        min_dist = float('inf')
        entry_edge_idx = -1
        
        for i in range(len(polygon_points)):
            edge = LineString([polygon_points[i], polygon_points[(i + 1) % len(polygon_points)]])
            dist = entry_point.distance(edge)
            if dist < min_dist:
                min_dist = dist
                entry_edge_idx = i
        
        # Find the opposite edge (approximately opposite to the entry edge)
        num_edges = len(polygon_points)

        if num_edges >= 5:
            if restrict_exit_to_opposite_edge:
                # Base opposite edge index
                base_opposite = (entry_edge_idx + num_edges // 2) % num_edges
                # Randomly choose between base, base+1, or base-1, avoiding adjacent edges
                # offset = np.random.choice([-1, 0, 1])
                offset = 0
                opposite_edge_idx = (base_opposite + offset) % num_edges
            else:
                # Randomly choose an edge
                opposite_edge_idx = np.random.randint(0, num_edges)
                # But it cannot be the same as the entry edge
                while opposite_edge_idx == entry_edge_idx:
                    opposite_edge_idx = np.random.randint(0, num_edges)
            # Check if the chosen edge is adjacent to entry edge
            # while (opposite_edge_idx == entry_edge_idx or 
            #        opposite_edge_idx == (entry_edge_idx + 1) % num_edges or 
            #        opposite_edge_idx == (entry_edge_idx - 1) % num_edges):
            #     offset = np.random.choice([-1, 0, 1])
            #     opposite_edge_idx = (base_opposite + offset) % num_edges
        else:
            # If there are less than 5 edges, then the opposite edge is the middle edge
            opposite_edge_idx = (entry_edge_idx + num_edges // 2) % num_edges
        
        # Get the opposite edge
        opposite_edge = LineString([
            polygon_points[opposite_edge_idx],
            polygon_points[(opposite_edge_idx + 1) % len(polygon_points)]
        ])
        
        # Find the midpoint of the opposite edge
        midpoint = opposite_edge.interpolate(0.5, normalized=True)
        
        # Get edge length for scaling the variance
        edge_length = opposite_edge.length
        
        # Sample random point along edge with normal distribution around midpoint
        # Using edge_length/6 as std dev to keep ~99.7% of points within the edge
        offset = np.random.normal(0, edge_length/5)
        
        # Create exit point by moving offset distance along the edge from midpoint
        exit_point = opposite_edge.interpolate(0.5 + offset/edge_length, normalized=True)
        exit_points.append((exit_point.x, exit_point.y))

        velocity = np.array([exit_point.x, exit_point.y]) - np.array([entry_point.x, entry_point.y])
        # Normalize velocity vector
        velocity = velocity / np.linalg.norm(velocity)

        velocity_vectors.append(velocity)

        
    return exit_points, velocity_vectors # the number of exit points is the same as the number of parents







def visualize_entry_points(polygon_points: List[Tuple[float, float]], offspring: List[Tuple[float, float]]):
    import matplotlib.pyplot as plt
    
    # Extract x and y coordinates
    x_offspring, y_offspring = zip(*offspring) if offspring else ([], [])

    # Plotting
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy

    plt.figure(figsize=(5, 3))  # Made slightly wider to accommodate legend
    plt.plot(x, y, 'k-', label='Polygon Boundary', alpha=0.5)
    plt.scatter(x_offspring, y_offspring, c='red', alpha=0.6, label='Offspring Points', s=3)
    plt.title('Thomas Point Process on Polygon Boundary')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    plt.show()


import matplotlib.pyplot as plt
def visualize_airspace(polygon: Tuple[np.ndarray, float], entry_points: List[Tuple[float, float]], exit_points: List[Tuple[float, float]], velocity_vectors: List[np.ndarray]):
    """
    Visualize the airspace polygon with entry points, exit points and velocity vectors.
    
    Args:
        polygon: Tuple containing polygon vertices and area
        entry_points: List of entry point coordinates
        exit_points: List of exit point coordinates 
        velocity_vectors: List of velocity vectors
    """
    
    # Create closed polygon by appending first vertex
    x = [v[0] for v in polygon]
    y = [v[1] for v in polygon]
    x.append(x[0])
    y.append(y[0])
    
    # Create plot
    plt.figure(figsize=(7,5))
    
    # Plot polygon boundary
    plt.plot(x, y, 'k-', linewidth=2, label='Airspace Boundary')

    # Convert entry_points, exit_points, and velocity_vectors to numpy arrays
    entry_points = np.array(entry_points)
    exit_points = np.array(exit_points)
    velocity_vectors = np.array(velocity_vectors)

    # Scatter plot the entry points
    plt.scatter(entry_points[:, 0], entry_points[:, 1], color='green', label='Entry Points')
    
    # Scatter plot the exit points
    plt.scatter(exit_points[:, 0], exit_points[:, 1], color='red', label='Exit Points')
    
    # Plot the velocity vectors
    # Plot velocity vectors as arrows from entry points
    for entry_point, velocity in zip(entry_points, velocity_vectors):
        plt.arrow(entry_point[0], entry_point[1], 
                 velocity[0]*50, velocity[1]*50,  # Scale velocity for visibility
                 head_width=10, head_length=10, fc='g', ec='g', 
                 alpha=0.5, length_includes_head=True)
        
    # Plot lines connecting entry and exit points
    for entry, exit in zip(entry_points, exit_points):
        plt.plot([entry[0], exit[0]], [entry[1], exit[1]], 
                 'b--', alpha=0.3, linewidth=1)
        
    # Add trajectory IDs with adjustable text positions to avoid overlap
    texts = []
    for i, (entry, exit) in enumerate(zip(entry_points, exit_points)):
        # Calculate midpoint of trajectory for initial text position
        mid_x = (entry[0] + exit[0]) / 2
        mid_y = (entry[1] + exit[1]) / 2
        
        # Add text with small offset from midpoint
        text = plt.text(mid_x, mid_y, str(i), 
                       fontsize=8,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0),
                       horizontalalignment='center',
                       verticalalignment='center')
        texts.append(text)
    
    # Try to prevent text overlap using adjustText if available
    try:
        from adjustText import adjust_text
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                   expand_points=(1.5, 1.5))
    except ImportError:
        # If adjustText not available, keep simple text labels
        print('adjustText not available, keeping simple text labels')
        pass
    

    plt.grid(True)
    plt.axis('equal')
    plt.title('Generated Airspace from Thomas-Jensen Process')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()



def assign_arrival_times_to_trajectories(entry_points: List[Tuple[float, float]], exit_points: List[Tuple[float, float]],
                                          arrivals: List[float], parent_indices: List[int],
                                          mode: str = 'uniform', seed: int = 69420) -> List[Tuple[float, float]]:
    if mode == 'uniform':
        # Shuffle the arrival times
        np.random.seed(seed)
        np.random.shuffle(arrivals)
        return arrivals
    else:
        raise ValueError(f"Invalid mode: {mode}. Please choose 'uniform'.")
    

