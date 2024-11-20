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
    len1 = np.linalg.norm(e1 - s1)
    len2 = np.linalg.norm(e2 - s2)
    
    if len1 >= len2:
        return s1, e1, s2, e2
    else:
        return s2, e2, s1, e1

def compute_segment_distance(s1, e1, s2, e2):
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

    attended_dv = dv + 10_000 if dv > 5 else dv

    attented_dtdv = np.exp(5 * dt) * attended_dv if dt > 0.1 else np.exp(dt) * attended_dv # dt: 0 -> 1, exp(dt) -> e

    return attented_dtdv

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


def compute_distance_matrix(segments):
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
            dist = compute_segment_distance(s1, e1, s2, e2)
            
            # Distance matrix is symmetric
            distance_matrix[i,j] = dist
            distance_matrix[j,i] = dist
            
    return distance_matrix
