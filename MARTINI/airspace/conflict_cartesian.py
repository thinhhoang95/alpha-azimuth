from MARTINI.definitions.trajectory import Trajectory 
import numpy as np
from itertools import combinations

LATERAL_DISTANCE_THRESHOLD = 6 * 1.852 # 6 nautical miles, in kilometers
VERTICAL_DISTANCE_THRESHOLD_LOWER = 1000 # 1000 feet below FL290
VERTICAL_DISTANCE_THRESHOLD_UPPER = 2000 # 2000 feet above FL290

def check_conflicts(trajectories: list[Trajectory],
                    vertical_distance_threshold_lower: float = VERTICAL_DISTANCE_THRESHOLD_LOWER,
                    vertical_distance_threshold_upper: float = VERTICAL_DISTANCE_THRESHOLD_UPPER,
                    lateral_distance_threshold: float = LATERAL_DISTANCE_THRESHOLD):
    """Check for conflicts between all pairs of trajectories.
    
    Args:
        trajectories: List of Trajectory objects to check for conflicts
        
    Returns:
        list: List of tuples (traj1_idx, traj2_idx, segment1_idx, segment2_idx) 
              indicating conflicting trajectory segments
    """
    conflicts = []
    
    # Check each pair of trajectories
    for (i1, traj1), (i2, traj2) in combinations(enumerate(trajectories), 2):
        # Check each segment pair
        for seg1 in range(len(traj1.waypoints) - 1):
            for seg2 in range(len(traj2.waypoints) - 1):
                # Get segment endpoints
                p1_start = traj1.waypoints[seg1]
                p1_end = traj1.waypoints[seg1 + 1]
                p2_start = traj2.waypoints[seg2]
                p2_end = traj2.waypoints[seg2 + 1]

                p1_start_xyz = traj1.waypoints_xyz[seg1]
                p1_end_xyz = traj1.waypoints_xyz[seg1 + 1]
                p2_start_xyz = traj2.waypoints_xyz[seg2]
                p2_end_xyz = traj2.waypoints_xyz[seg2 + 1]

                # Get passing times at segment endpoints
                t1_start = traj1.passing_time[seg1]
                t1_end = traj1.passing_time[seg1 + 1]
                t2_start = traj2.passing_time[seg2]
                t2_end = traj2.passing_time[seg2 + 1]
                
                # Get altitudes at segment endpoints
                alt1_start = traj1.altitudes[seg1]
                alt1_end = traj1.altitudes[seg1 + 1]
                alt2_start = traj2.altitudes[seg2]
                alt2_end = traj2.altitudes[seg2 + 1]
                
                # Check if segments are in conflict
                in_conflict, t_lb = segments_in_conflict_cartesian(p1_start_xyz, p1_end_xyz, alt1_start, alt1_end,
                                     p2_start_xyz, p2_end_xyz, alt2_start, alt2_end,
                                     t1_start, t1_end, t2_start, t2_end,
                                     vertical_distance_threshold_lower, vertical_distance_threshold_upper,
                                     lateral_distance_threshold)
                if in_conflict:
                    conflicts.append((i1, i2, seg1, seg2, t_lb))
    
    return conflicts


def segments_in_conflict_cartesian(p1_start_xyz, p1_end_xyz, alt1_start, alt1_end,
                             p2_start_xyz, p2_end_xyz, alt2_start, alt2_end,
                             t1_start, t1_end, t2_start, t2_end,
                             vertical_distance_threshold_lower: float = VERTICAL_DISTANCE_THRESHOLD_LOWER,
                             vertical_distance_threshold_upper: float = VERTICAL_DISTANCE_THRESHOLD_UPPER,
                             lateral_distance_threshold: float = LATERAL_DISTANCE_THRESHOLD):
    
    # Check if altitudes are within the vertical distance threshold
    if abs(alt1_start - alt2_start) > vertical_distance_threshold_lower or \
       abs(alt1_end - alt2_end) > vertical_distance_threshold_lower:
        return False, np.nan
    
    # Rename variables for clarity
    x1 = p1_start_xyz
    x3 = p2_start_xyz
    x2 = p1_end_xyz
    x4 = p2_end_xyz

    t1 = t1_start
    t2 = t1_end
    t3 = t2_start
    t4 = t2_end


    
    a = x1 - x3 - t1 * (x2 - x1) / (t2 - t1) + t3 * (x4 - x3) / (t4 - t3)
    b = (x2 - x1) / (t2 - t1) - (x4 - x3) / (t4 - t3)

    # Intersection time heuristic bounds must be between the start and end times of the segments
    t_min = min(t1, t2, t3, t4)
    t_max = max(t1, t2, t3, t4)

    # Intersection time bound assuming that ||x1(t) - x3(t)|| <= LATERAL_DISTANCE_THRESHOLD
    delta = (a.dot(b))**2 - b.dot(b) * (a.dot(a) - lateral_distance_threshold**2)

    if delta < 0:
        return False, np.nan
    if b.dot(b) <= 1e-6:
        if a.dot(a) - lateral_distance_threshold**2 > 0:
            return False, np.nan # Non-overlapping segments
        else:
            return True, np.nan # Parallel and overlapping segments
    t_lb = (-a.dot(b) - np.sqrt(delta)) / b.dot(b)
    t_ub = (-a.dot(b) + np.sqrt(delta)) / b.dot(b)

    # Check if t_lb and t_ub are within the segment bounds
    if t_lb > t1 and t_lb < t2 and t_lb > t3 and t_lb < t4 and \
       t_ub > t1 and t_ub < t2 and t_ub > t3 and t_ub < t4:
        return True, t_lb
    
    return False, np.nan

# Deprecated
def _segments_in_conflict(p1_start, p1_end, alt1_start, alt1_end,
                        p2_start, p2_end, alt2_start, alt2_end,
                        vertical_distance_threshold_lower: float = VERTICAL_DISTANCE_THRESHOLD_LOWER,
                        vertical_distance_threshold_upper: float = VERTICAL_DISTANCE_THRESHOLD_UPPER,
                        lateral_distance_threshold: float = LATERAL_DISTANCE_THRESHOLD):
    """Check if two trajectory segments are in conflict.
    
    Args:
        p1_start, p1_end: Start and end points of first segment (lat, lon)
        alt1_start, alt1_end: Start and end altitudes of first segment
        p2_start, p2_end: Start and end points of second segment (lat, lon)
        alt2_start, alt2_end: Start and end altitudes of second segment
        
    Returns:
        bool: True if segments are in conflict, False otherwise
    """
    # Create interpolation points (using 10 points per segment)
    t = np.linspace(0, 1, 10)
    
    # Interpolate positions
    p1_interp = np.array([np.interp(t, [0, 1], [p1_start[0], p1_end[0]]),  # lat
                         np.interp(t, [0, 1], [p1_start[1], p1_end[1]])])   # lon
    p2_interp = np.array([np.interp(t, [0, 1], [p2_start[0], p2_end[0]]),  # lat
                         np.interp(t, [0, 1], [p2_start[1], p2_end[1]])])   # lon
    
    # Interpolate altitudes
    alt1_interp = np.interp(t, [0, 1], [alt1_start, alt1_end])
    alt2_interp = np.interp(t, [0, 1], [alt2_start, alt2_end])
    
    # Check each interpolated point pair
    for i in range(len(t)):
        for j in range(len(t)):
            # Calculate lateral distance (using approximate formula)
            lat1, lon1 = p1_interp[:, i]
            lat2, lon2 = p2_interp[:, j]
            
            # Convert degrees to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            lateral_distance = 6371 * c  # Earth radius in km
            
            # Calculate vertical separation
            vertical_separation = abs(alt1_interp[i] - alt2_interp[j])
            
            # Check if distances violate thresholds
            if (lateral_distance < lateral_distance_threshold and 
                vertical_separation < vertical_distance_threshold_lower):
                return True
                
    return False
    