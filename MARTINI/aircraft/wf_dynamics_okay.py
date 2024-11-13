"""
Wind-free Aircraft Dynamics

This module computes the aircraft dynamics in a wind-free environment. The aircraft is assumed to be a point mass with no wind effects. The aircraft dynamics are based on the equations of motion in the body frame.
The following assumptions were used:
- Throttle, speed brake, flaps and slats are sufficient to maintain both the aircraft speed and vertical speed.
- The aircraft is guided laterally in heading mode.
- The gravitational acceleration is g0 

Author: Thinh Hoang
Based on: Derivation of a Point-Mass Aircraft Model used for Fast-Time Simulation, MITRE Corporation, 2015
Date: 2024-05-07
"""

import numpy as np

# ******************************************
# * Physical constants                     *
# ******************************************
g0 = 9.80665 # m/s**2

# ******************************************
# * Controller gains                       *
# ******************************************

kvp = -0.05
kvi = 0

khp = -0.05
khi = 0
khd = -0.5

kgammap = -0.1

kphip = -5.1
kphii = 0

# Violent Roll Control to Correct Heading Error
kpsip = -5.85
kpsii = 0

kp_x = 0.1
kp_y = 0.1

def sat(x: float, x_min: float, x_max: float) -> float:
    """Saturation function

    Args:
        x (float): value
        x_min (float): minimum of value
        x_max (float): maximum of value

    Returns:
        float: saturated value of x between x_min and x_max
    """
    return np.minimum(np.maximum(x, x_min), x_max)

def wind_free_system(t, s: np.array, u: np.array,
              kvp: float, kvi: float, 
              khp: float, khi: float,
              kphip: float, kphii: float,
              kpsip: float, kpsii: float) -> np.array:
    """A closed loop control system model for the point-mass aircraft in wind-free condition,
    constant gravitational acceleration

    Args:
        s (np.array): aircraft's state x, y, h, V, gamma, psi, phi, evi, ehi, epsii, ephii (11 states)
        u (np.array): commanded values for psi_c, Vc, hc (3 inputs) for control signal
        kvp (float): P gain for thrust velocity control
        kvi (float): I gain for thrust velocity control
        khp (float): P gain for altitude control
        khi (float): I gain for altitude control
        kphip (float): P gain for roll angle control
        kphii (float): I gain for roll angle control
        kpsip (float): P gain for heading angle control
        kpsii (float): I gain for heading angle control

    Returns:
        np.array: time derivative of the states
    """
    
    # Unpacking the state variables
    # Dynamics variables
    x = s[0] # x position (m)
    y = s[1] # y position (m)
    h = s[2] # altitude (m)
    V = s[3] # ground speed (m/s)
    gamma = s[4] # flight path angle (rad)
    psi = s[5] # heading angle (rad)
    # Inner loop state variables
    phi = s[6] # roll angle (rad)
    # T = s[7] # thrust (N), not needed since V_dot serves its purpose
    
    # Error integral variables 
    evi = s[7] # integral of ground speed tracking error
    ehi = s[8] # integral of altitude tracking error 
    epsii = s[9] # integral of heading tracking error
    ephii = s[10] # integral of roll angle tracking error    
    
    # Unpacking the control inputs
    # Tc = u[0] # commanded thrust (N)
    psi_c = u[0] # commanded heading angle (rad)
    Vc = u[1] # commanded speed (m/s)
    hc = u[2] # commanded altitude (m)
    
    # State-space model for wind-free condition
    dsdt = np.zeros(14)
    dsdt[0] = V * np.cos(gamma) * np.cos(psi) # x_dot
    dsdt[1] = V * np.cos(gamma) * np.sin(psi) # y_dot
    dsdt[2] = -V * np.sin(gamma) # h_dot
    # gamma positive -> h decreases
    
    # Outer control logic for thrust and pitch, corresponding to SPD mode
    target_accel = kvp * (V - Vc) + kvi * evi # PI controller using thrust
    dsdt[3] = sat(target_accel, -0.25, 0.25) # this is a rough bound for the deceleration
    # more accurate performance model maybe required
    
    # We also assume that speed brake is effective enough so both the speed control and vertical rate can be achieved
    target_vs = sat(khp * (h - hc) + khi * ehi + khd * (-V * np.sin(gamma)), -9.144, 9.144) # PI controller to determine the target vertical speed, 1800fpm maximum
    target_gamma = sat(np.arcsin( -target_vs / V ), -10 * np.pi / 180, 10 * np.pi/180) # PI controller using flight path angle, gamma_dot
    gamma_dot = kgammap * (gamma - target_gamma)
    dsdt[4] = gamma_dot
    
    # TODO: Implement speed control using pitch
    
    # Computation of lift from flight path angle
    # L = (-gamma_dot + g0 * np.cos(gamma) / V) * (m * V / np.cos(phi))
    
    # Outer control logic for heading, corresponding to HDG mode
    dsdt[5] = - (g0 * np.cos(gamma) / V - gamma_dot) * np.tan(phi)/np.cos(gamma) # psi_dot
    # Ensure no 360 degree jumps
    if (psi - psi_c) > np.pi:
        psi_c += 2 * np.pi
    elif (psi - psi_c) < -np.pi:
        psi_c -= 2 * np.pi
    target_phi = - (kpsip * (psi - psi_c) + kpsii * epsii) # PI controller to determine the roll angle
    dsdt[6] = sat(kphip * (phi - target_phi) + kphii * ephii, -0.436, 0.436, -0.001, 0.001)  # PI controller for the heading angle, phi_dot with maximum 25 degree bank angle

    # Error integrators
    dsdt[7] = V - Vc # ev integration
    dsdt[8] = h - hc # eh integration
    dsdt[9] = psi - psi_c # epsi integration
    dsdt[10] = phi - target_phi # ephi integration
    
    # Constant commanded inputs
    dsdt[11] = 0 # psi_c
    dsdt[12] = 0 # V_c
    dsdt[13] = 0 # hc
    
    return dsdt 



def heading_to_geometric_angle(heading_angle: float) -> float:
    """Convert from heading angle (clockwise from north) to geometric angle (counter-clockwise from east).
    
    Args:
        heading_angle: Angle in radians, measured clockwise from north (0-2π)
        
    Returns:
        float: Geometric angle in radians, measured counter-clockwise from east (0-2π)
    """
    # First convert heading to east-referenced clockwise
    east_referenced = (heading_angle - np.pi/2) 
    
    # Then flip direction to counter-clockwise
    geometric = -east_referenced
    
    # Normalize to 0-2π range
    while geometric < 0:
        geometric += 2*np.pi
    while geometric >= 2*np.pi:
        geometric -= 2*np.pi
        
    return geometric

def geometric_angle_to_heading(geometric_angle: float) -> float:
    """Convert from geometric angle (counter-clockwise from east) to heading angle (clockwise from north).
    
    Args:
        geometric_angle: Angle in radians, measured counter-clockwise from east (0-2π)
        
    Returns:
        float: Heading angle in radians, measured clockwise from north (0-2π)
    """
    # First flip direction to clockwise
    east_referenced = -geometric_angle
    
    # Then convert to north-referenced
    heading = east_referenced + np.pi/2
    
    # Normalize to 0-2π range
    while heading < 0:
        heading += 2*np.pi
    while heading >= 2*np.pi:
        heading -= 2*np.pi
        
    return heading




def wind_free_system_with_lateral_control(t, s: np.array, u: np.array,
              kvp: float, kvi: float, 
              khp: float, khi: float,
              kphip: float, kphii: float,
              kpsip: float, kpsii: float) -> np.array:
    """A closed loop control system model for the point-mass aircraft in wind-free condition,
    constant gravitational acceleration. Lateral position control is added.

    Args:
        s (np.array): aircraft's state x, y, h, V, gamma, psi, phi, evi, ehi, epsii, ephii (11 states)
        u (np.array): commanded values for psi_c, Vc, hc, xc, yc (5 inputs) for vectoring
        kvp (float): P gain for thrust velocity control
        kvi (float): I gain for thrust velocity control
        khp (float): P gain for altitude control
        khi (float): I gain for altitude control
        kphip (float): P gain for roll angle control
        kphii (float): I gain for roll angle control
        kpsip (float): P gain for heading angle control
        kpsii (float): I gain for heading angle control
        ki_x (float): I gain for lateral position control
        ki_y (float): I gain for lateral position control

    Returns:
        np.array: time derivative of the states
    """
    
    # Unpacking the state variables
    # Dynamics variables
    x = s[0] # x position (m)
    y = s[1] # y position (m)
    h = s[2] # altitude (m)
    V = s[3] # ground speed (m/s)
    gamma = s[4] # flight path angle (rad)
    psi = s[5] # heading angle (rad)
    # Inner loop state variables
    phi = s[6] # roll angle (rad)
    # T = s[7] # thrust (N), not needed since V_dot serves its purpose
    
    # Error integral variables 
    evi = s[7] # integral of ground speed tracking error
    ehi = s[8] # integral of altitude tracking error 
    epsii = s[9] # integral of heading tracking error
    ephii = s[10] # integral of roll angle tracking error    
    
    # Unpacking the control inputs
    # Tc = u[0] # commanded thrust (N)
    Vc = u[0] # commanded speed (m/s)
    hc = u[1] # commanded altitude (m)
    xc = u[2] # commanded x position (m)
    yc = u[3] # commanded y position (m)
    
    # State-space model for wind-free condition
    dsdt = np.zeros(15)
    psi_geometric = heading_to_geometric_angle(psi) # Geometric angle
    dsdt[0] = V * np.cos(gamma) * np.cos(psi_geometric) # x_dot
    dsdt[1] = V * np.cos(gamma) * np.sin(psi_geometric) # y_dot
    dsdt[2] = -V * np.sin(gamma) # h_dot
    # gamma positive -> h decreases
    
    # Outer control logic for thrust and pitch, corresponding to SPD mode
    target_accel = kvp * (V - Vc) + kvi * evi # PI controller using thrust
    dsdt[3] = sat(target_accel, -0.25, 0.25) # this is a rough bound for the deceleration
    # more accurate performance model maybe required
    
    # We also assume that speed brake is effective enough so both the speed control and vertical rate can be achieved
    target_vs = sat(khp * (h - hc) + khi * ehi + khd * (-V * np.sin(gamma)), -9.144, 9.144) # PI controller to determine the target vertical speed, 1800fpm maximum
    target_gamma = sat(np.arcsin( -target_vs / V ), -10 * np.pi / 180, 10 * np.pi/180) # PI controller using flight path angle, gamma_dot
    gamma_dot = kgammap * (gamma - target_gamma)
    dsdt[4] = gamma_dot
    
    # TODO: Implement speed control using pitch
    
    # Computation of lift from flight path angle
    # L = (-gamma_dot + g0 * np.cos(gamma) / V) * (m * V / np.cos(phi))
    
    # Outer control logic for heading, corresponding to HDG mode
    dsdt[5] = - (g0 * np.cos(gamma) / V - gamma_dot) * np.tan(phi)/np.cos(gamma) # psi_dot

    # Lateral position control for phi_c
    # Compute heading angle to target position
    dx = xc - x
    dy = yc - y
    target_psi_geometric = np.arctan2(dy, dx)
    
    # Convert to heading angle (clockwise from north)
    target_psi = geometric_angle_to_heading(target_psi_geometric)
        
    # P controller for heading
    psi_c = target_psi

    # print(f"psi: {psi}, psi_c: {psi_c}")
    
    # Ensure no 360 degree jumps
    if (psi - psi_c) > np.pi:
        psi_c += 2 * np.pi
    elif (psi - psi_c) < -np.pi:
        psi_c -= 2 * np.pi
        
    target_phi = - (kpsip * (psi - psi_c) + kpsii * epsii) # PI controller to determine the roll angle
    dsdt[6] = sat(kphip * (phi - target_phi) + kphii * ephii, -0.436, 0.436)  # PI controller for the heading angle, phi_dot with maximum 25 degree bank angle
    
    # Error integrators
    dsdt[7] = V - Vc # ev integration
    dsdt[8] = h - hc # eh integration
    dsdt[9] = psi - psi_c # epsi integration
    dsdt[10] = phi - target_phi # ephi integration
    
    # Constant commanded inputs
    dsdt[11] = 0 # V_c
    dsdt[12] = 0 # hc
    dsdt[13] = 0 # xc
    dsdt[14] = 0 # yc
    
    return dsdt 






def ode_wrapper_for_system(t, x):
    # Decomposing x into s, u
    s = x[0:11] # x[0] to x[10]
    u = x[11:14] # x[11] to x[13]
    
    
    dsdt = wind_free_system(t, s, u, kvp, kvi, khp, khi, kphip, kphii, kpsip, kpsii)
    return dsdt

def ode_wrapper_for_system_with_lateral_control(t, x):
    # Decomposing x into s, u
    s = x[0:11] # x[0] to x[10]
    u = x[11:16] # x[11] to x[15]
    
    dsdt = wind_free_system_with_lateral_control(t, s, u, kvp, kvi, khp, khi, kphip, kphii, kpsip, kpsii)
    return dsdt