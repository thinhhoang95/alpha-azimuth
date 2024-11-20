import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from functools import partial

from .trajclus_features import compute_distance_matrix

def perform_clustering(X, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.01,
                       algo='hdbscan'):
    """Perform HDBSCAN clustering with custom metric support"""

    # Obtain the distance matrix
    distance_matrix = compute_distance_matrix(X)

    if algo == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method='eom',
            metric='precomputed'
        )
    elif algo == 'spectral':
        raise NotImplementedError('Spectral clustering is not implemented yet')
    else:
        raise ValueError(f'Unknown clustering algorithm: {algo}')
        
    cluster_labels = clusterer.fit_predict(distance_matrix)
    
    return cluster_labels, clusterer
