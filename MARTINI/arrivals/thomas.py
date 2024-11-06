import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

import numpy as np
from shapely.geometry import Polygon, LineString, Point

from MARTINI.airspace import geo


def thomas_point_process_border(points: List[Tuple[float, float]], lambda_parent: float, mu: float, sigma: float, random_state: int = None, min_dist_to_vertex: float = 50, min_transit_length: float = 50) -> Tuple[List[Tuple[float, float]], List[int], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    Generate random points along the border of a polygon using the Thomas Point Process.

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

