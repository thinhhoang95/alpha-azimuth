import igraph as ig
import leidenalg
import numpy as np

def leiden_community_detection(adjacency_matrix, resolution=1.0, n_iterations=2):
    """
    Perform Leiden community detection on a network.
    
    Parameters:
    -----------
    adjacency_matrix : numpy.ndarray
        The adjacency matrix of the network
    resolution : float
        Resolution parameter for the Leiden algorithm (default=1.0)
    n_iterations : int
        Number of iterations of the Leiden algorithm (default=2)
        
    Returns:
    --------
    communities : list
        List of community assignments for each node
    quality : float
        Quality of the partition (modularity)
    """
    # Convert adjacency matrix to igraph
    sources, targets = np.where(adjacency_matrix > 0)
    edges = list(zip(sources.tolist(), targets.tolist()))
    weights = adjacency_matrix[adjacency_matrix > 0]
    
    # Create igraph object
    G = ig.Graph(edges=edges, directed=False)
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=n_iterations
    )
    
    return list(partition.membership), partition.quality()

# Example usage
if __name__ == "__main__":
    # Create a simple example network
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    
    # Detect communities
    communities, quality = leiden_community_detection(adj_matrix)
    
    print(f"Found communities: {communities}")
    print(f"Partition quality: {quality}")
