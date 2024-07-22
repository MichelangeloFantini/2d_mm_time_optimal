import numpy as np

from ..velocity_generation import compute_velocities_in_cone_2d
from ..trajectory_computations import compute_all_trajectories_multi_pose


def calculate_trajectory(self, points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min,vertex_angle_deg=180,magnitude_step=4, angle_step=20, loop=False, sampling_rate=3):

    int_points = self.generate_intermediate_points_withouth_trajectory(points, prediction_horizon=prediction_horizon)

    
    params, waypoints_poses, goals = self.compute_positions(starting_configurations, int_points)
    if prediction_horizon == None:
        points.append(points[0])

    max_vel_mag = np.linalg.norm(v_max)
    dict_graph = self.create_graph(waypoints_poses, goals, sampling_rate=sampling_rate)
    end_effector_velocities = compute_velocities_in_cone_2d(points, 0, max_vel_mag, vertex_angle_deg=vertex_angle_deg, magnitude_step=magnitude_step, angle_step=angle_step, loop=loop, prediction_horizon=prediction_horizon)

    q_dot_dict = self.generate_waypoints_q_dot(dict_graph, end_effector_velocities, v_max, v_min)

    results, dict_res, shortest = compute_all_trajectories_multi_pose(dict_graph, q_dot_dict, a_max, a_min, v_max, v_min, prediction_horizon=prediction_horizon, analyze=False, remove_bad=False)
    return results, dict_res, shortest