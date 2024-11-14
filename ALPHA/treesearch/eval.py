import numpy as np
from ..obstacle.obstacle import Obstacle
from MARTINI.airspace.geo import line_segment_intersects_polygon
from MARTINI.aircraft.flight_plan import FlightPlan
from typing import Any


def eval_plan(flight_plan: FlightPlan, obstacles: Obstacle, airspace: Any, wind_field: Any = None):
    if wind_field is not None:
        raise NotImplementedError('Wind field evaluation is not implemented yet')
    
    # Get trajectory 4D from flight plan
    trajectory_4d = flight_plan.get_trajectory_4d()

    time_cost = trajectory_4d[0][-1] # t, x, y, z
    obstacle_hits = 0

    # Check each segment of the trajectory
    for i in range(len(trajectory_4d[0]) - 1):
        start = np.array([trajectory_4d[1][i], trajectory_4d[1][i]])
        end = np.array([trajectory_4d[2][i + 1], trajectory_4d[2][i + 1]])

        # Check if the segment intersects with any obstacle
        for obstacle in obstacles.get_polygons():
            if line_segment_intersects_polygon(start, end, obstacle):
                obstacle_hits += 1

    return time_cost, obstacle_hits
