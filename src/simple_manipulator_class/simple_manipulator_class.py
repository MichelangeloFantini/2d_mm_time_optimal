import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import casadi as ca

from src.point_mass_trajectory_optimization import space_curve, velocity_curve


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a 3D rotation matrix (SO(3)).
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), np.sin(roll)],
                    [0, -np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                    [0, 1, 0],
                    [np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), np.sin(yaw), 0],
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combine the rotation matrices around x, y, and z axes
    R = R_z @ R_y @ R_x
    return R

class SimpleManipulator:
    def __init__(self):
        self.name = 'SimpleManipulator'
        
        self.base_width = 0.4  # Width of the base (rectangle)
        self.base_length = 1.0  # Length of the base (rectangle)
        self.base_y = 0.5  # Y coordinate of the base center
        self.base_x = 0.5

        self.arm_length1 = 1.0  # Length of the first arm
        self.arm_width1 = 0.05  # Width of the first arm
        self.arm_length2 = 1  # Length of the second arm
        self.arm_width2 = 0.05  # Width of the second arm
        self.interval = 50

        self.m2 = 2
        self.m3 = 2
        from . import plotting
        setattr(SimpleManipulator, 'plot_dynamic', plotting.plot_dynamic)
        setattr(SimpleManipulator, 'plot_static', plotting.plot_static)
        setattr(SimpleManipulator, 'plot_dynamic_params', plotting.plot_dynamic_params)
        setattr(SimpleManipulator, 'plot_end_effector', plotting.plot_end_effector)
        setattr(SimpleManipulator, 'plot_end_effector_best', plotting.plot_end_effector_best)
        setattr(SimpleManipulator, 'plot_end_effector_and_velocities', plotting.plot_end_effector_and_velocities)
        setattr(SimpleManipulator, 'plot_end_effector_best_trajectory_and_velocity', plotting.plot_end_effector_best_trajectory_and_velocity)
        from . import inverse_kynematics
        setattr(SimpleManipulator, 'inverse_kynematics_dynamic', inverse_kynematics.inverse_kynematics_dynamic)
        setattr(SimpleManipulator, 'base_movement', inverse_kynematics.base_movement)
        setattr(SimpleManipulator, 'q_dot_time', inverse_kynematics.q_dot_time)
        setattr(SimpleManipulator, 'compute_positions', inverse_kynematics.compute_positions)
        setattr(SimpleManipulator, 'compute_error', inverse_kynematics.compute_error)
        setattr(SimpleManipulator, 'new_pose_taylor', inverse_kynematics.new_pose_taylor)
        setattr(SimpleManipulator, 'new_pose_luoponov', inverse_kynematics.new_pose_luoponov)
        setattr(SimpleManipulator, 'new_pose_luoponov_control', inverse_kynematics.new_pose_luoponov_control)
        setattr(SimpleManipulator, 'converge_to_goal', inverse_kynematics.converge_to_goal)
        from . import end_effector
        setattr(SimpleManipulator, 'displace_pose', end_effector.displace_pose)
        setattr(SimpleManipulator, 'generate_intermediate_points_withouth_trajectory', end_effector.generate_intermediate_points_withouth_trajectory)
        from . import trajectory_graph
        setattr(SimpleManipulator, 'create_graph', trajectory_graph.create_graph)
        from . import q_velocity_generation
        setattr(SimpleManipulator, 'generate_waypoints_q_dot', q_velocity_generation.generate_waypoints_q_dot)
        setattr(SimpleManipulator, 'check_velocity_in_bound', q_velocity_generation.check_velocity_in_bound)
        from . import motion_model
        setattr(SimpleManipulator, 'robot_motion_model', motion_model.robot_motion_model)
        setattr(SimpleManipulator, 'new_motion_model', motion_model.new_motion_model)
        from . import trajectory_computation
        setattr(SimpleManipulator, 'calculate_trajectory', trajectory_computation.calculate_trajectory)


    def compute_link_positions(self, theta1, theta2): 
        TI_0 = np.block([[np.eye(3), np.array([[self.base_x], [self.base_y], [0]])], [0, 0, 0, 1]])
        T01 = np.block([[euler_to_rotation_matrix(0, 0, -theta1), np.array([[self.base_length/2], [self.base_width/2], [0]])], [0, 0, 0, 1]])
        T12 = np.block([[euler_to_rotation_matrix(0, 0, -theta2), np.array([[self.arm_length1], [0], [0]])], [0, 0, 0, 1]])
        T23 = np.block([[euler_to_rotation_matrix(0, 0, 0), np.array([[self.arm_length2], [0], [0]])], [0, 0, 0, 1]]) # end effector
        TI1 = TI_0 @ T01
        x1, y1 = TI1[0, 3], TI1[1, 3]
        TI2 = TI1 @ T12
        x2, y2 = TI2[0, 3], TI2[1, 3]
        TI3 = TI2 @ T23
        x3, y3 = TI3[0, 3], TI3[1, 3]
        return x1, y1, x2, y2, x3, y3
    
    def compute_jacobian(self, theta1, theta2):
        J = np.array([[-self.arm_length1*np.sin(theta1) - self.arm_length2*np.sin(theta1 + theta2), -self.arm_length2*np.sin(theta1 + theta2)],
                    [self.arm_length1*np.cos(theta1) + self.arm_length2*np.cos(theta1 + theta2), self.arm_length2*np.cos(theta1 + theta2)]])
        return J

    def process_results(self, dict_res, shortest):
        results = []
        times = []
        results.append(dict_res[shortest[0]][shortest[1]])
        times.append(results[0][0][6])
        for i in range(1, len(shortest)-1):
            results.append(dict_res[shortest[i]][shortest[i+1]])
            times.append(results[i][0][6] + times[i-1])
        return results, times
    
    def plot(self, dict_res, shortest, dynamics=True):
        if dynamics:
            self.plot_dynamic(dict_res, shortest)
        else:
            self.plot_static(dict_res, shortest)

    def forward_kinematics_model(self):
        theta1 = ca.MX.sym('theta1')
        theta2 = ca.MX.sym('theta2')

        q = ca.vertcat(theta1, theta2)
        T_I_0 = ca.vertcat(
            ca.horzcat(1, 0, 0, self.base_x),
            ca.horzcat(0, 1, 0, self.base_y),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1)
        )
        T_0_1 = ca.vertcat(
            ca.horzcat(ca.cos(theta1), -ca.sin(theta1), 0, self.base_length/2),
            ca.horzcat(ca.sin(theta1), ca.cos(theta1), 0, self.base_width/2),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1)
        )
        T_1_2 = ca.vertcat(
            ca.horzcat(ca.cos(theta2), -ca.sin(theta2), 0, self.arm_length1),
            ca.horzcat(ca.sin(theta2), ca.cos(theta2), 0, 0),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1)
        )
        T_2_3 = ca.vertcat(
            ca.horzcat(1, 0, 0, self.arm_length2),
            ca.horzcat(0, 1, 0, 0),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1)
        )
        T_I_3 = T_I_0 @ T_0_1 @ T_1_2 @ T_2_3
        return ca.Function('forward_kinematics', [q], [T_I_3])
    
    def end_effector_pose_func(self):
        # Interested in x, y coordinates of end effector expressed in the inertial frame, this corresponds 
        # to the first two elements of the last column of the transformation matrix from 3 to I
        model = self.forward_kinematics_model()
        theta1 = ca.MX.sym('theta1')
        theta2 = ca.MX.sym('theta2')
        q = ca.vertcat(theta1, theta2)

        x = self.base_x + self.base_length/2 + self.arm_length1*ca.cos(theta1) + self.arm_length2*ca.cos(theta1 + theta2)
        y = self.base_y + self.base_width/2 + self.arm_length1*ca.sin(theta1) + self.arm_length2*ca.sin(theta1 + theta2)
        ee = ca.vertcat(x, y)
        T_I_3 = model(q)
        return ca.Function('end_effector_pose_func', [q], [ee])

