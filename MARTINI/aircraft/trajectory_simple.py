import numpy as np

def get_4d_trajectory_from_flight_plan(entry_time, entry_psi, spd_ref, alt_ref, xy_ref, wind_field = None):
    if wind_field is not None:
        raise NotImplementedError("Wind field not implemented yet")
        
    # Store the trajectory
    marks_x = []
    marks_y = []
    marks_h = []
    marks_t = []

    # Convert the spd_ref to m/s
    spd_ref = np.array(spd_ref) * 0.514444444

    clock = entry_time

    # We iterate through the waypoints in the trajectory
    wp_x, wp_y = xy_ref[0]
    wp_z, wp_v = alt_ref[0], spd_ref[0]

    # Add the first waypoint to the trajectory
    marks_t.append(clock)
    marks_h.append(wp_z)
    marks_x.append(wp_x)
    marks_y.append(wp_y)

    # We assume the altitude and speed values apply for the segment FOLLOWING the current waypoint
    for (x,y), z, v in zip(xy_ref[1:], alt_ref[1:], spd_ref[1:]):
        # Distance to the next waypoint
        x_diff = x - wp_x
        y_diff = y - wp_y
        dist_to_next_wp = np.sqrt(x_diff**2 + y_diff**2)
        
        # Time to travel this distance at the current speed
        time_to_next_wp = dist_to_next_wp / wp_v

        # If we climb or descend at 1800 ft/min, the maximum altitude change we can achieve in this segment is
        z_diff_max = time_to_next_wp * (1800 / 60) # ft

        # The actual desired altitude change
        z_diff = z - wp_z

        if np.abs(z_diff) > 0:
            if np.abs(z_diff) > z_diff_max:
                # We will continue climbing or descending through the next waypoint
                z_diff = z_diff_max * np.sign(z_diff)
            else:
                # Find the level off point in this segment
                t_level_off = time_to_next_wp * (np.abs(z_diff) / z_diff_max)
                # Add the level off point to the trajectory
                marks_t.append(clock + t_level_off)
                marks_h.append(wp_z + z_diff)
                marks_x.append(wp_x + x_diff * t_level_off / time_to_next_wp)
                marks_y.append(wp_y + y_diff * t_level_off / time_to_next_wp)

        # Add the next waypoint to the trajectory
        marks_t.append(clock + time_to_next_wp)
        marks_h.append(wp_z + z_diff)
        marks_x.append(x)
        marks_y.append(y)

        clock += time_to_next_wp

        wp_x, wp_y = x, y # nearest waypoint behind the aircraft
        wp_z, wp_v = z, v # altitude and speed at the nearest waypoint behind the aircraft
        
    return np.array(marks_t), np.array(marks_x), np.array(marks_y), np.array(marks_h)