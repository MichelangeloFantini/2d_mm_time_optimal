import numpy as np

from ..velocity_generation import compute_velocities_in_cone_2d
from ..trajectory_computations import compute_all_trajectories

def calculate_trajectory(self, points, prediction_horizon, a_max, a_min, v_max, v_min,vertex_angle_deg=180,magnitude_step=4, angle_step=20, loop=False):
    module_min = 0
    # norm of vmax
    module_max = np.linalg.norm([*v_max])
    df = compute_velocities_in_cone_2d(points, module_min, module_max,vertex_angle_deg=vertex_angle_deg, angle_step=angle_step, magnitude_step=magnitude_step, prediction_horizon=prediction_horizon, loop=loop)
    results, dict_res, shortest = compute_all_trajectories(points,df, a_max, a_min, v_max, v_min, prediction_horizon=prediction_horizon)
    return results, dict_res, shortest