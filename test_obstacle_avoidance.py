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
from src.obstacles.circles import circle_obstacle


if __name__ == '__main__':
    mobile_robot = MobileManipulator()
    points = [(2,1.7), (10,1), (15,2.1)]
    starting_configurations = (0.5, np.pi/2, -np.pi/2)
    prediction_horizon = len(points)-1
    a_max = (20,20,20)
    a_min = (-20,-20,-20)
    max_vel = 1
    v_max = (max_vel,max_vel,max_vel)
    v_min = (-max_vel,-max_vel,-max_vel)
    N = 200

    obstacles = [circle_obstacle(3, 2, 0.5), circle_obstacle(8.5, 2, 0.5)]
    # obstacles = [circle_obstacle(5, 2, 0.5)]
    # obstacles = []
    motion_model = mobile_robot.robot_motion_model()
    end_effector_pose_func = mobile_robot.end_effector_pose_func()

    # solve the problem cpc
    start_time = time.time()
    X0_no_tn, tf = robotic_arm_initial_guess(mobile_robot, points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, N, position_noise=0.1)
    X0 = ca.vertcat(tf, *X0_no_tn)
    X, total_dim = optimize(mobile_robot, points, v_max, v_min, a_max, a_min, prediction_horizon, X0, np.inf, motion_model, end_effector_pose_func, N=N, d_tol=0.01, constrain_final_point=False, obstacles=obstacles)
    print('time taken by casadi:', time.time()-start_time)
    # solve problem with sequential optimization
    start_time = time.time()
    X0_no_tn, ts, Ns = robotic_arm_initial_guess(mobile_robot, points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, N, position_noise=0, is_sequential_guess=True)
    X0 = ca.vertcat(*ts, *X0_no_tn)
    X_seq, total_dim_seq = optimize_sequential(mobile_robot, points, v_max, v_min, a_max, a_min, prediction_horizon, X0, motion_model, end_effector_pose_func, Ns=Ns, d_tol=0.01, obstacles=obstacles)
    print('time taken by casadi sequential:', time.time()-start_time)

    qs_seq, q_dots_seq, us_seq = decompose_X(X_seq, 3, total_dim_seq)
    zipped_list_seq = [i for i in zip(*qs_seq)]

    qs, q_dots, us = decompose_X(X, 3, total_dim)
    zipped_list = [i for i in zip(*qs)]
    # mobile_robot.plot_dynamic(starting_configurations, dict_res, shortest, points)
    mobile_robot.plot_dynamic_params(zipped_list_seq, interval = (float(X_seq[0]/(N*1000))), points=points, obstacles=obstacles, circle_decomposition=True)

    mobile_robot.plot_dynamic_params(zipped_list, interval = (float(X[0]/(N*1000))), points=points, obstacles=obstacles, circle_decomposition=True)

    #plot end effector pose
    compare_trajectories_casadi_plot([X_seq], points, None, None, end_effector_pose_func, q_size = len(v_max), state_dim=[total_dim], labels=['casadi cpc'])
