import numpy as np

from .mobile_manipulator_class import MobileManipulator
from ..point_mass_trajectory_optimization import space_curve, velocity_curve

def generate_intermediate_points(self, results, dict_res, shortest, points, time_interval=0.1):
    ''' Generate intermediate points for the shortest path and plot the dynamic trajectory of the mobile manipulator. '''
    results, times = self.process_results(dict_res, shortest)
    # print(f'Total time taken before optimization: {times[-1]} seconds')
    # Generate intermediate points every time_interval seconds
    intermediate_points = []
    for i in range(len(times)):
        a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, tf_x, ts_x, ts_1_x = results[i][0]
        a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, tf_y, ts_y, ts_1_y = results[i][1]

        time_points = np.arange(0, tf_x, time_interval) 
        x = [space_curve(t, a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in time_points]
        y = [space_curve(t, a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, ts_y, ts_1_y) for t in time_points]
        vx = [velocity_curve(t, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in time_points]
        vy = [velocity_curve(t, a1_y, a2_y, a3_y, a4_y, a5_y, ts_y, ts_1_y) for t in time_points]
        # create vector to determine if point is the first point
        is_waypoint = np.zeros(len(time_points))
        if i == 0:
            is_waypoint[0] = 1
        if i!=0:
            time_points += times[i-1]
        is_waypoint[-1] = 1
        # create tuple of (x, y, vx, vy, t, encoded) for each time point
        intermediate_points.extend(list(zip(x, y, vx, vy, time_points, is_waypoint)))
    return intermediate_points

def generate_intermediate_points_withouth_trajectory(self, points, prediction_horizon=None):
    intermediate_points = []

    if prediction_horizon == None:
        prediction_horizon = len(points)
    for i in range(prediction_horizon+1):
        if i == len(points):
            intermediate_points.append((points[0][0], points[0][1], 0, 0, 0, 1))
        else:
            intermediate_points.append((points[i][0], points[i][1], 0, 0, 0, 1))
    return intermediate_points

def displace_pose(self, pose, goal, x_increment=0.1, tolerance=0.05):
    ''' Displace a pose by x_increment and y_increment. '''
    # Move the base position and compute the new end effector position
    to_right = []
    to_left = []
    # Determine x limits for the base
    i = 0

    e_left_term = 0
    e_right_term = 0
    # Get the initial end effector position to convert to the goal
    e, J = self.compute_error(pose, goal)
    # print('initial e:', np.linalg.norm(e), 'goal:', goal)
    previous_left = pose
    previous_right = pose

    while np.linalg.norm(e_left_term) < tolerance or np.linalg.norm(e_right_term) < tolerance:
        x_left = previous_left[0] - x_increment
        x_right = previous_right[0] + x_increment
        # Compute the new end effector position
        e_left, J_left = self.compute_error((x_left, previous_left[1], previous_left[2]), goal)
        e_right, J_right = self.compute_error((x_right, previous_right[1], previous_right[2]), goal)
        # Compute the new end effector position
        # Taylor series
        theta1_left, theta2_left = self.new_pose_taylor((x_left, previous_left[1], previous_left[2]), e_left, J_left)
        theta1_right, theta2_right = self.new_pose_taylor((x_right, previous_right[1], previous_right[2]), e_right, J_right)

        # Lyapunov while
        # theta1_left, theta2_left = self.new_pose_luoponov_control((x_left, previous_left[1], previous_left[2]), goal)
        # theta1_right, theta2_right = self.new_pose_luoponov_control((x_right, previous_right[1], previous_right[2]), goal)
        # if theta1_left == None:
        #     theta1_left, theta2_left = previous_left[1], previous_left[2]
        # if theta1_right == None:
        #     theta1_right, theta2_right = previous_right[1], previous_right[2]

        # Lyapunov time based
        # theta1_left, theta2_left = self.new_pose_luoponov((x_left, previous_left[1], previous_left[2]), e_left, J_left, goal)
        # theta1_right, theta2_right = self.new_pose_luoponov((x_right, previous_right[1], previous_right[2]), e_right, J_right, goal)  

        # Check if the end effector is within the workspace
        e_left_term, _ = self.compute_error((x_left, theta1_left, theta2_left), goal)
        e_right_term, _ = self.compute_error((x_right, theta1_right, theta2_right), goal)
        if np.linalg.norm(e_left_term) < tolerance:
            to_left = [(x_left, theta1_left, theta2_left)] + to_left
        if np.linalg.norm(e_right_term) < tolerance:
            to_right.append((x_right, theta1_right, theta2_right))

        previous_left = (x_left, theta1_left, theta2_left)
        previous_right = (x_right, theta1_right, theta2_right)
        i += 1
    poses = to_left + [pose] + to_right
    return poses