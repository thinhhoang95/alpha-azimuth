from wf_dynamics import ode_wrapper_for_system_with_lateral_control
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
MAX_SOLVER_TIME = 3 * 3600 # 3 hours



def plot_state_vector(sol):
    # Plot all the fundamental states
    plt.figure(figsize=(10,9))
    plt.title('Simulated Aircraft States')
    
    plt.subplot(3,3,1)
    plt.plot(sol.t, sol.y[0,:]) # x
    plt.title('x')
    
    plt.subplot(3,3,2)
    plt.plot(sol.t, sol.y[1,:]) # y
    plt.title('y')
    
    plt.subplot(3,3,3)
    plt.plot(sol.t, sol.y[2,:]) # h
    plt.title('Alt')
    
    plt.subplot(3,3,4)
    plt.plot(sol.t, sol.y[3,:]) # V
    plt.title('GS')
    
    plt.subplot(3,3,5)
    plt.plot(sol.t, sol.y[4,:]) # gamma 
    plt.title('FPA')
    
    plt.subplot(3,3,6)
    plt.plot(sol.t, sol.y[5,:]) # psi
    plt.title('HDG')
    
    plt.subplot(3,3,7)
    plt.plot(sol.t, sol.y[6,:]) # phi
    plt.title('Roll')
    
    # plt.tight_layout()
    
    plt.show()



def get_4d_trajectory_from_flight_plan(entry_time, entry_psi, spd_ref, alt_ref, xy_ref, wind_field = None):
    if wind_field is not None:
        raise NotImplementedError("Wind field not implemented yet")
    
    clock = entry_time # in seconds
    max_clock = entry_time + 2 * 3600 # 2 hours

    # Global state vector
    s = np.zeros(15)
    # Populating the initial state vector with the entry point conditions
    s[3] = spd_ref[0] * 1852 / 3600 # speed
    s[2] = alt_ref[0] * 0.3048 # altitude
    s[0] = xy_ref[0][0] # x position
    s[1] = xy_ref[0][1] # y position
    s[5] = entry_psi * np.pi / 180 # heading

    # Record the conditions at the waypoints
    event_times = []
    event_states = []

    event_times.append(clock)
    event_states.append(s)

    # Record the state vector
    state_times = []
    state_states = []


    # Skip first waypoint since we're already there, then iterate through remaining waypoints
    wp_idx = 0
    for (x,y), z, v in zip(xy_ref[1:], alt_ref[1:], spd_ref[1:]):
        wp_idx += 1
        print(f"Directing to waypoint {wp_idx}: ({x}, {y})")
        # Prepare the reference vectors for the upcoming waypoint 
        dsdt = ode_wrapper_for_system_with_lateral_control
        # Initial conditions
        y0 = np.copy(s) 
        # Speed
        y0[11] = v * 1852 / 3600
        # Altitude
        y0[12] = z * 0.3048
        # Position
        y0[13] = x
        y0[14] = y

        # Solve the ODE
        sol = solve_ivp(dsdt, t_span=[0, MAX_SOLVER_TIME], y0=y0)

        t_vec = sol.t
        s_vec = sol.y

        x_vec = s_vec[0,:]
        y_vec = s_vec[1,:]
        
        # Find when aircraft first enters waypoint capture radius
        r_wp = 1000  # meters, typical waypoint capture radius
        dist_to_wp = np.sqrt((x_vec - x)**2 + (y_vec - y)**2)
        wp_reached_indices = np.where(dist_to_wp <= r_wp)[0]
        
        if len(wp_reached_indices) == 0:
            plot_state_vector(sol)
            print(f"Integration stopped at waypoint {wp_idx}")
            print(f"Last x value: ", x_vec[-1])
            print(f"Last y value: ", y_vec[-1])
            raise RuntimeError(f"Aircraft never reached waypoint ({x}, {y})")
            
        # Get first time waypoint was reached
        wp_reached_idx = wp_reached_indices[0]
        wp_reached_time = t_vec[wp_reached_idx]
        
        # Update state vector to waypoint reached position
        s = sol.y[:, wp_reached_idx]
        

        # Record the event
        event_times.append(clock)
        event_states.append(s)

        # Record the state vector
        # Resample the state vector at 1 Hz
        t_resampled = np.arange(0, wp_reached_time, 1)
        interpolator = interp1d(t_vec, s_vec, axis=1)
        s_resampled = interpolator(t_resampled)
        state_times.append(t_resampled + clock)
        state_states.append(s_resampled)

        clock += wp_reached_time
        
        if clock > max_clock:
            raise RuntimeError("Maximum simulation time exceeded")
        
    return event_times, event_states, state_times, state_states

def test_get_4d_trajectory_from_flight_plan():
    entry_time = 0
    entry_psi = 90
    spd_ref = [250, 250, 250, 250]
    alt_ref = [25000, 25000, 25000, 25000]
    xy_ref = [[0,0], [50000,0], [50000,50000], [0,50000]]
    event_times, event_states, state_times, state_states = get_4d_trajectory_from_flight_plan(entry_time, entry_psi, spd_ref, alt_ref, xy_ref)

    return event_times, event_states, state_times, state_states

if __name__ == "__main__":
    test_get_4d_trajectory_from_flight_plan()