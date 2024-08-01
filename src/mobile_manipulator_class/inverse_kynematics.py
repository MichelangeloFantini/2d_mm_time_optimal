import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from .mobile_manipulator_class import MobileManipulator

def inverse_kynematics_dynamic(self, current_end_effector, current_params, future_goals, k =1):
    ''' Compute the inverse kinematics for the mobile manipulator. '''
    # Compute the inverse kinematics for the mobile manipulator
    time_interval = future_goals[0][4] - current_end_effector[4]

    # x = current_params[0]+current_end_effector[2]*time_interval
    # x = future_goals[0][0] - 0.5
    # print('x:', x)
    # x, y = self.base_movement(current_end_effector, future_goals)

    # x = 0.5
    # print('x:', x, 'y:', y)
    # First determine the base movement

    error_params = (current_params[0], current_params[1], current_params[2])
    # e, J  = self.compute_error(error_params, future_goals[0])
    # Compute the joint velocities
    # goal_q_dot_ee = np.array([[future_goals[0][2]], [future_goals[0][3]]])
    # Compute the new joint angles
    # theta1, theta2 = self.new_pose_taylor(current_params, e, J)
    # theta1, theta2 = self.new_pose_luoponov(current_params, e, J, future_goals[0], time_interval, k)
    # x, theta1, theta2 = self.converge_to_goal(error_params, future_goals[0], t=0.1)
    x, theta1, theta2 = self.new_pose_luoponov_control(error_params, future_goals[0], t=0.1)

    return x, theta1, theta2

def new_pose_taylor(self, current_params, e, J):
    J_inv = np.linalg.inv(J)
    q = J_inv @ e
    theta1 = current_params[1] + q[0, 0]
    theta2 = current_params[2] + q[1, 0]
    return theta1, theta2

def new_pose_luoponov(self, current_params, e, J, goal, time_interval=0, k=1):
    goal_q_dot_ee = np.array([[goal[2]], [goal[3]]])
    goal_pose_ee = np.array([goal[0], goal[1]])
    J_inv = np.linalg.inv(J)
    q_dot = J_inv @ (goal_q_dot_ee + k*e)
    t = self.q_dot_time(q_dot, current_params, np.array([[goal[0]], [goal[1]]]), time_interval)
    if t == None:
        return current_params[1], current_params[2]
    theta1 = current_params[1] + q_dot[0, 0]*t
    theta2 = current_params[2] + q_dot[1, 0]*t
    return theta1, theta2

def new_pose_luoponov_control(self, current_params, goal, t=0.1, max_iter=100):
    e, J = self.compute_error(current_params, goal)
    x, theta1, theta2 = current_params
    goal_q_dot_ee = np.array([[goal[2]], [goal[3]]])

    count = 0
    params = [(x, theta1, theta2)]
    while np.linalg.norm(e) > 0.01 and count < max_iter:
        J_inv = np.linalg.pinv(J)
        q_dot = J_inv @ (goal_q_dot_ee + e)
        x = x + q_dot[0, 0]*t
        theta1 = theta1 + q_dot[1, 0]*t
        theta2 = theta2 + q_dot[2, 0]*t
        e, J = self.compute_error((x, theta1, theta2), goal)
        count += 1
    if count == max_iter:
        return None, None, None
    return x, theta1, theta2

def converge_to_goal(self, current_params, goal, t=1, max_iter=5):
    count =0
    x, theta1, theta2 = current_params
    while count < max_iter:
        cur_pose = (x, theta1, theta2)
        theta1, theta2 = self.new_pose_luoponov_control(cur_pose, goal, t)
        if theta1 != None:
            return x, theta1, theta2
        count += 1
        # move arm closer to the goal
        if x < goal[0]:
            x += 0.2
        else:
            x -= 0.2
        theta1 = current_params[1]
        theta2 = current_params[2]
    return current_params


def compute_error(self, current_params, future_goal):
    ''' Compute the error for the inverse kinematics. '''
    x0, y0, x1, y1, x2, y2, x3, y3 = self.compute_link_positions(current_params[0], current_params[1], current_params[2])
    # Compute Jacobian
    J = self.compute_jacobian_whole(current_params)
    # compute inverse of Jacobian
    e = np.array([[future_goal[0] - x3], [future_goal[1] - y3]])
    return e, J


def base_movement(self, current_end_effector, future_goals, reach=2):
    ''' Determine the base movement for the mobile manipulator. '''
    xc = ca.MX.sym('xc')
    yc = ca.MX.sym('yc')
    U = ca.vertcat(xc, yc)
    g = []
    obj = 0
    opts = {'print_time': False, 'ipopt.sb': 'yes', 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.print_level': 1, 'ipopt.constr_viol_tol': 1e-6, 'ipopt.acceptable_constr_viol_tol': 1e-8, 'ipopt.acceptable_iter': 0}
    lbx = ca.vertcat(-np.inf, self.base_y)
    ubx = ca.vertcat(np.inf, self.base_y)
    args = {'lbx': lbx, 'ubx': ubx, 'lbg': 0, 'ubg': reach**2}
    x0 = ca.vertcat(current_end_effector[0], self.base_y)
    for i, goal in enumerate(future_goals):
        goal = ca.vertcat(goal[0], goal[1])
        obj = obj + (U - goal).T @ (U - goal)
        g.append((U - goal).T @ (U - goal))
        nlp = {'x': U, 'f': obj, 'g': ca.vertcat(*g), 'p': []}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        args['x0'] = x0
        res = solver(**args)
        res = res['x']
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            break
        # plt.show()
        solution = (float(res[0]), float(res[1]))
    return solution

def q_dot_time(self, q_dot, current_params, goal_ee, time_guess):
    t = ca.MX.sym('t')
    x = ca.vertcat(t)
    # rotation using current params
    X = ca.MX.sym('X', 2)
    P = ca.MX.sym('P', 12) 

    X_list = []
    X_list.append(P[1] + P[-2]*t)
    X_list.append(P[2] + P[-1]*t)
    X = ca.horzcat(*X_list)

    R1 = ca.vertcat(
        ca.horzcat(ca.cos(X[0]), -ca.sin(X[0])),
        ca.horzcat(ca.sin(X[0]),  ca.cos(X[0]))
    )
    R2 = ca.vertcat(
        ca.horzcat(ca.cos(X[1]), -ca.sin(X[1])),
        ca.horzcat(ca.sin(X[1]),  ca.cos(X[1]))
    )
    R = R1 @ R2

    rhs = ca.vertcat(P[0] + P[3], P[4] + P[5]) + R1 @ ca.vertcat(P[6], 0) + R @ ca.vertcat(P[7], 0)

    f = ca.Function('f', [t, P], [rhs])

    obj = (ca.vertcat(P[8], P[9]) - f(t, P)).T @ (ca.vertcat(P[8], P[9]) - f(t, P))
    opts = {'print_time': False, 'ipopt.sb': 'yes', 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-8, 'ipopt.print_level': 1, 'ipopt.constr_viol_tol': 1e-6, 'ipopt.acceptable_constr_viol_tol': 1e-8}
    

    args = {'lbx': 0, 'ubx': np.inf, 'lbg': 0, 'ubg': 0}
    nlp = {'x': x, 'f': obj, 'g': [], 'p': P}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    args['x0'] = ca.vertcat(time_guess)
    args['p'] = ca.vertcat(current_params[0], current_params[1], current_params[2], self.base_length/2, self.base_width/2, self.base_y, self.arm_length1, self.arm_length2, goal_ee[0], goal_ee[1], q_dot[0], q_dot[1])
    res = solver(**args)
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        return None
    return float(res['x'])


def compute_positions(self, starting_configurations, intermediate_points):
    configurations = []
    goals = [intermediate_points[0]]
    waypoints_configs = [(starting_configurations[0], starting_configurations[1], starting_configurations[2])]
    x, theta1, theta2 = starting_configurations
    for i in range(1,len(intermediate_points)):
        configurations.append((x, theta1, theta2))
        x, theta1, theta2 = self.inverse_kynematics_dynamic(intermediate_points[i-1],(x, theta1, theta2), intermediate_points[i:])
        if intermediate_points[i][5] == 1:
            waypoints_configs.append((x, theta1, theta2))
            goals.append(intermediate_points[i])
    return configurations, waypoints_configs, goals