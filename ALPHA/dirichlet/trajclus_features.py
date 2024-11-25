# We follow the proposed features in the paper: Trajectory Clustering: A Partition-and-Group Framework
# by Jae-Gil Lee, Jiawei Han, Kyu-Young Whang
# Link: https://hanj.cs.illinois.edu/pdf/sigmod07_jglee.pdf

import numpy as np

def get_projection_distances(si, ei, sj, ej):
    """
    Calculate projection distances between two line segments.
    
    Args:
        si, ei: Start and end points of the first (longer) line segment (2D numpy arrays)
        sj, ej: Start and end points of the second (shorter) line segment (2D numpy arrays)
    
    Returns:
        lv1: Distance from sj to its projection on segment (si,ei)
        lv2: Distance from ej to its projection on segment (si,ei) 
        ls1: Distance from si to projection of sj
        ls2: Distance from projection of ej to ei
    """
    # Vector of the first line segment
    vi = ei - si
    vi_unit = vi / np.linalg.norm(vi)
    
    # Get projections of sj and ej onto line containing (si,ei)
    vs = sj - si  # Vector from si to sj
    ve = ej - si  # Vector from si to ej
    
    # Calculate projection points
    proj_s_dist = np.dot(vs, vi_unit)  # Scalar projection
    proj_e_dist = np.dot(ve, vi_unit)  # Scalar projection
    
    ps = si + proj_s_dist * vi_unit  # Vector projection (point)
    pe = si + proj_e_dist * vi_unit  # Vector projection (point)
    
    # Calculate vertical distances (perpendicular to first segment)
    lv1 = np.linalg.norm(sj - ps)
    lv2 = np.linalg.norm(ej - pe)
    
    # Calculate distances along the first segment
    ls1 = np.linalg.norm(si - ps)
    ls2 = np.linalg.norm(pe - ei)

    # Calculate angle between segments
    vj = ej - sj  # Vector of second segment
    cos_theta = np.dot(vi_unit, vj / np.linalg.norm(vj))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors

    # Calculate orientation distance (perpendicular component of second segment)
    # lt = np.linalg.norm(ej - sj) * np.sin(theta)
    lt = np.sin(theta)
    
    return lv1, lv2, ls1, ls2, lt

def compute_features(si, ei, sj, ej):
    """
    Compute geometric features between two line segments with length normalization.
    
    Args:
        si, ei: Start and end points of the first (longer) line segment (2D numpy arrays)
        sj, ej: Start and end points of the second (shorter) line segment (2D numpy arrays)
    
    Returns:
        dv: Normalized vertical distance feature
        ds: Normalized longitudinal distance feature
        dt: Normalized orientation distance feature
    """
    lv1, lv2, ls1, ls2, lt = get_projection_distances(si, ei, sj, ej)
    
    # Get segment lengths for normalization
    len_i = np.linalg.norm(ei - si)  # Length of longer segment
    len_j = np.linalg.norm(ej - sj)  # Length of shorter segment
    
    # Normalize features by appropriate length scales
    dv = (lv1 ** 2 + lv2 ** 2) #  / (lv1 + lv2)
    ds = np.minimum(ls1, ls2)
    dt = lt

    return dv, ds, dt

def get_ordered_segments(s1, e1, s2, e2):
    """
    Orders two line segments by length, returning endpoints of longer segment first.
    
    Args:
        s1, e1: Start and end points of first line segment (2D numpy arrays)
        s2, e2: Start and end points of second line segment (2D numpy arrays)
        
    Returns:
        si, ei: Start and end points of longer segment
        sj, ej: Start and end points of shorter segment
    """
    # Convert inputs to numpy arrays if they aren't already
    s1 = np.array(s1) if not isinstance(s1, np.ndarray) else s1
    e1 = np.array(e1) if not isinstance(e1, np.ndarray) else e1
    s2 = np.array(s2) if not isinstance(s2, np.ndarray) else s2
    e2 = np.array(e2) if not isinstance(e2, np.ndarray) else e2
    len1 = np.linalg.norm(e1 - s1)
    len2 = np.linalg.norm(e2 - s2)
    
    if len1 >= len2:
        return s1, e1, s2, e2
    else:
        return s2, e2, s1, e1

def compute_segment_distance(s1, e1, s2, e2, allow_cutoff = False,
                             max_distance = 10.0, max_dt = 0.1,
                             verbose = False):
    """
    Compute the distance between two line segments using geometric features.
    
    Args:
        s1, e1: Start and end points of first line segment (2D numpy arrays)
        s2, e2: Start and end points of second line segment (2D numpy arrays)
        
    Returns:
        float: Combined distance measure between the segments
    """
    # Order segments by length
    si, ei, sj, ej = get_ordered_segments(s1, e1, s2, e2)
    
    # Compute geometric distance features
    dv, ds, dt = compute_features(si, ei, sj, ej)

    if verbose:
        print(f"dv: {dv}, ds: {ds}, dt: {dt}")

    if allow_cutoff:
        if max_distance is None or max_dt is None:
            raise ValueError("max_distance and max_dt must be provided if allow_cutoff is True")
        attended_dv = np.inf if dv > max_distance else dv
        if verbose:
            if dv > max_distance:
                print(f"attended_dv is inf because dv > max_distance")
        attended_dtdv = np.inf if dt > max_dt else np.exp(dt) * attended_dv
        if verbose:
            if dt > max_dt:
                print(f"attended_dtdv is inf because dt > max_dt")
        return attended_dtdv
    else:
        attended_dv = dv
        attended_dtdv = np.exp(dt) * dv
        return attended_dtdv

    # Weights for the features
    # weights = [1, 0, 100]
    
    # return np.sum(weights * np.array([dv, ds, dt])) # dv: vertical distance, ds: longitudinal distance, dt: orientation distance

def compute_segment_features_auto(s1, e1, s2, e2):
    """
    Compute geometric features for two line segments, ordering them by length.

    This function takes the start and end points of two line segments, orders them by their lengths,
    and computes geometric distance features between the segments. The features computed include:
    - dv: Vertical distance between the segments
    - ds: Longitudinal distance between the segments
    - dt: Orientation distance between the segments

    Args:
        s1 (np.ndarray): Start point of the first line segment (2D numpy array).
        e1 (np.ndarray): End point of the first line segment (2D numpy array).
        s2 (np.ndarray): Start point of the second line segment (2D numpy array).
        e2 (np.ndarray): End point of the second line segment (2D numpy array).

    Returns:
        tuple: A tuple containing three elements:
            - dv (float): Vertical distance feature.
            - ds (float): Longitudinal distance feature.
            - dt (float): Orientation distance feature.
    """
    # Order segments by length
    si, ei, sj, ej = get_ordered_segments(s1, e1, s2, e2)
    
    # Compute geometric distance features
    dv, ds, dt = compute_features(si, ei, sj, ej)

    return dv, ds, dt


def compute_distance_matrix(segments, allow_cutoff = False,
                            max_distance = 10.0, max_dt = 0.1):
    """
    Compute pairwise distance matrix between line segments using geometric features.
    
    Args:
        segments: List of line segments, where each segment is a tuple (start_point, end_point)
                 and points are 2D numpy arrays
        
    Returns:
        distance_matrix: Square numpy array containing pairwise distances
    """
    n = len(segments)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        s1, e1 = segments[i]
        for j in range(i+1, n):
            s2, e2 = segments[j]
            
            # Compute distance between segments i and j
            dist = compute_segment_distance(s1, e1, s2, e2,
                                            allow_cutoff=allow_cutoff,
                                            max_distance=max_distance,
                                            max_dt=max_dt)
            
            # Distance matrix is symmetric
            distance_matrix[i,j] = dist
            distance_matrix[j,i] = dist
            
    return distance_matrix



def get_adjacency_matrix_from_distance_matrix_by_thresholding(distance_matrix: np.ndarray, max_distance = 400):
    """
    Convert a distance matrix to an adjacency matrix using a Gaussian kernel transformation.

    This function performs the following steps:
    1. Replaces infinite values in the distance matrix with zeros
    2. Computes sigma as the mean of non-zero distances
    3. Applies a Gaussian kernel transformation to convert distances to similarities
    
    Args:
        distance_matrix (np.ndarray): A square symmetric matrix containing pairwise distances
            between elements. Can contain inf values to indicate disconnected elements.

    Returns:
        np.ndarray: A square symmetric adjacency matrix where each element represents
            the similarity between pairs of elements. Values are in range [0,1],
            where 1 indicates maximum similarity and 0 indicates no connection.

    Note:
        The Gaussian kernel transformation uses the formula:
        similarity = exp(-distance²/(2σ²))
        where σ is computed as the mean of all non-zero distances
    """
    # Replace inf values in distance matrix with zeros in similarity calculation
    similarity_matrix = np.copy(distance_matrix)
    similarity_matrix[np.isinf(similarity_matrix)] = 0


    nonzero_mask = similarity_matrix != 0
    similarity_matrix[nonzero_mask] = np.where(similarity_matrix[nonzero_mask] < max_distance, 1, 0)
    return similarity_matrix


    # # Convert distance matrix to similarity matrix using Gaussian kernel
    # sigma = np.mean(similarity_matrix[similarity_matrix > 0])  # Can be tuned based on your needs
    # # Only apply exponential transformation to non-zero values
    # nonzero_mask = similarity_matrix != 0
    # similarity_matrix[nonzero_mask] = np.exp(-similarity_matrix[nonzero_mask]**2 / (2 * sigma**2))
    # adjacency_matrix = similarity_matrix

    # return adjacency_matrix