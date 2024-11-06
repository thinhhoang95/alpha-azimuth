"""
ISA - International Standard Atmosphere

This module provides functions to calculate atmospheric properties based on the International Standard Atmosphere (ISA) model.
The expressions limit to the troposphere only.

===============================================================
====>   ATTENTION: THIS FILE IS CURRENTLY NOT USED IN THE CODEBASE.     <====
===============================================================

Author: Thinh Hoang
Based on: REVISION OF ATMOSPHERE MODEL IN BADA AIRCRAFT PERFORMANCE MODEL, EEC TECHNICAL REPORT 2010-001
Date: 2024-05-07
"""

# Constants
EARTH_RADIUS = 6325766.0 # m (mean radius of the Earth)
g0 = 9.80665 # m/s^2 (standard gravity)
p0 = 101325.0 # Pa (pressure at sea level)
rho0 = 1.225 # kg/m^3 (density at sea level)
T0 = 288.15 # K (temperature at sea level)
a0 = 340.294 # m/s (speed of sound at sea level)

def geopotential_alt(altitude: float) -> float:
    """Calculate geopotential altitude from geometric altitude.

    Args:
        altitude (float): Geometric altitude in meters.

    Returns:
        float: Geopotential altitude in meters.
    """
    return EARTH_RADIUS * altitude / (EARTH_RADIUS + altitude)

def gravitational_accel(altitude: float) -> float:
    """Calculate gravitational acceleration at a given altitude.

    Args:
        altitude (float): Geometric altitude in meters.

    Returns:
        float: Gravitational acceleration in m/s^2.
    """
    return g0 * (EARTH_RADIUS / (EARTH_RADIUS + altitude))**2

def get_temperature(altitude: float) -> float:
    """Calculate temperature at a given altitude.

    Args:
        altitude (float): Geometric altitude in meters.

    Returns:
        float: Temperature in Kelvin.
    """
    return T0 - 0.0065 * geopotential_alt(altitude)

def get_pressure(altitude: float) -> float:
    """Calculate pressure at a given altitude.

    Args:
        altitude (float): Geometric altitude in meters.

    Returns:
        float: Pressure in Pascals.
    """
    return p0 * (1 - 0.0065 * geopotential_alt(altitude) / T0)**(g0 / (287.05 * 0.0065))

def get_air_density(altitude: float) -> float:
    """Calculate air density at a given altitude.

    Args:
        altitude (float): Geometric altitude in meters.

    Returns:
        float: Air density in kg/m^3.
    """
    return rho0 * (1 - 0.0065 * geopotential_alt(altitude) / T0)**((g0 / (287.05 * 0.0065)) - 1)

