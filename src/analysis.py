import numpy as np

from .point_mass_trajectory_optimization import space_curve, velocity_curve
from .plotting import plot_single_trajectory


def analyze_single_trajectory(params, start_point, end_point, starting_velocity, ending_velocity, threshold=0.1):
    ''' Given a single trajectory, analyze the trajectory and plot it if it does not meet the space constraints.'''
    final_point = np.zeros(len(params))
    final_velocity = np.zeros(len(params))
    for i in range(len(params)):
        a0, a1, a2, a3, a4, a5, tf, ts, ts_1 = params[i]
        final_point[i] = space_curve(tf, a0, a1, a2, a3, a4, a5, ts, ts_1)
        final_velocity[i] = velocity_curve(tf, a1, a2, a3, a4, a5, ts, ts_1)
    if np.linalg.norm(final_point - end_point) > threshold or np.linalg.norm(final_velocity - ending_velocity) > threshold:
        print("trajectory is bad")
        print("supposed end point: ", end_point)
        print("supposed end velocity: ", ending_velocity)
        print("computed end point: ", final_point)
        print("computed end velocity: ", final_velocity)
        plot_single_trajectory(params)

def remove_trajectory(x_result, y_result, start_point, end_point, threshold=0.2):
    ''' Given a single trajectory, return True if the trajectory does not meet the space constraints.'''
    final_point = np.zeros(len(params))
    final_velocity = np.zeros(len(params))
    for i in range(len(params)):
        a0, a1, a2, a3, a4, a5, tf, ts, ts_1 = params[i]
        final_point[i] = space_curve(tf, a0, a1, a2, a3, a4, a5, ts, ts_1)
        final_velocity[i] = velocity_curve(tf, a1, a2, a3, a4, a5, ts, ts_1)
    if np.linalg.norm(final_point - end_point) > threshold or np.linalg.norm(final_velocity - ending_velocity) > threshold:
        return True
    return False