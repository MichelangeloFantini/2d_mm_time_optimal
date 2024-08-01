import numpy as np
import casadi as ca
import time

from src.mobile_manipulator_class.mobile_manipulator_class import MobileManipulator
from src.casadi_initial_guess import robotic_arm_initial_guess, compute_tf
from src.casadi_optimize import optimize
from src.casadi_optimize_opti import optimize_opti
from src.decompose_casadi_solution import decompose_X
from src.plotting import compare_trajectories_casadi_plot
from src.casadi_sequential_tasks import optimize_sequential


if __name__ == '__main__':
    mobile_robot = MobileManipulator()
    points = [(2,1.7), (2.1,2.5), (2.2, 2),(10,2.1), (10, 2), (5, 2)]
    starting_configurations = (0.5, np.pi/2, -np.pi/2)
    prediction_horizon = len(points)-1
    a_max = (20,20,20)
    a_min = (-20,-20,-20)
    max_vel = 1
    v_max = (max_vel,max_vel,max_vel)
    v_min = (-max_vel,-max_vel,-max_vel)
    N = 200

    # point mass trajectory optimization
    start_time = time.time()
    magnitude_step = 4
    angle_step = 4
    vertex_angle_deg = 180
    results, dict_res, shortest = mobile_robot.calculate_trajectory(points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, magnitude_step=magnitude_step, angle_step=angle_step, vertex_angle_deg=vertex_angle_deg, loop=False, sampling_rate=10)
    print('time taken by sampling algorithm:', time.time()-start_time)
    tf = compute_tf(shortest, dict_res)
    mobile_robot.plot_dynamic(starting_configurations, dict_res, shortest, points)
    # solve the problem cpc
    start_time = time.time()
    X0_no_tn, tf = robotic_arm_initial_guess(mobile_robot, points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, N, position_noise=0.1)
    # X0_no_tn = [0]*len(X0_no_tn)
    # X0_no_tn[:len(starting_configurations)] = list(starting_configurations)
    X0 = ca.vertcat(tf, *X0_no_tn)
    motion_model = mobile_robot.robot_motion_model()
    end_effector_pose_func = mobile_robot.end_effector_pose_func()
    X, total_dim = optimize(points, v_max, v_min, a_max, a_min, prediction_horizon, X0, np.inf, motion_model, end_effector_pose_func, N=N, d_tol=0.01, constrain_final_point=True)
    print('time taken by casadi:', time.time()-start_time)
    # solve problem with sequential optimization
    start_time = time.time()
    X0_no_tn, ts, Ns = robotic_arm_initial_guess(mobile_robot, points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, N, position_noise=0, is_sequential_guess=True)
    # X0_no_tn = [0]*len(X0_no_tn)
    # X0_no_tn[:len(starting_configurations)] = list(starting_configurations)
    X0 = ca.vertcat(*ts, *X0_no_tn)
    X_seq, total_dim_seq = optimize_sequential(points, v_max, v_min, a_max, a_min, prediction_horizon, X0, motion_model, end_effector_pose_func, Ns=Ns, d_tol=0.01)
    print('time taken by casadi sequential:', time.time()-start_time)


    print('casadi solution cpc:', X[0], 'sampling algorithm:', tf, 'casadi sequential:', X_seq[0])

    qs, q_dots, us = decompose_X(X, 3, total_dim)
    zipped_list = [i for i in zip(*qs)]
    mobile_robot.plot_dynamic(starting_configurations, dict_res, shortest, points)
    mobile_robot.plot_dynamic_params(zipped_list, interval = (float(X_seq[0]/(N*1000))), points=points)

    #plot end effector pose
    compare_trajectories_casadi_plot([X, X_seq], points, dict_res, shortest, end_effector_pose_func, q_size = len(v_max), state_dim=[total_dim, total_dim_seq], labels=['casadi cpc', 'casadi sequential'])
