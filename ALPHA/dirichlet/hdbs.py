import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from functools import partial

from .trajclus_features import compute_distance_matrix

def perform_clustering(X, min_cluster_size=5, min_samples=None, metric='euclidean', 
                      custom_metric=None, weights=None):
    """Perform HDBSCAN clustering with custom metric support"""

    # Obtain the distance matrix
    distance_matrix = compute_distance_matrix(X)
        
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.1,
        cluster_selection_method='eom',
        metric='precomputed'
    )
        
    cluster_labels = clusterer.fit_predict(distance_matrix)
    
    return cluster_labels, clusterer

