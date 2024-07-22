import numpy as np  

from .point_mass_trajectory_optimization import space_curve, velocity_curve, acceleration_curve

def robotic_arm_initial_guess(mobile_robot, points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, N, d_tol=0.01, lambda_num_cols=1, analyze=False, remove_bad=False, position_noise=0.01, velocity_noise=0, acceleration_noise=0, is_sequential_guess=False):
    if mobile_robot.name == 'PointMass':
        results, dict_res, shortest = mobile_robot.calculate_trajectory(points, prediction_horizon, a_max, a_min, v_max, v_min, magnitude_step=1)
    else:
        results, dict_res, shortest = mobile_robot.calculate_trajectory(points, starting_configurations, prediction_horizon, a_max, a_min, v_max, v_min, magnitude_step=1)
    Ns, tf = compute_Ns(shortest, dict_res, N)
    ts = []
    for i in range(len(shortest)-1):
        t = dict_res[shortest[i]][shortest[i+1]][0][6]
        ts.append(t)
    X0_no_tn = []   

    for i in range(len(shortest)-1):
        # qs, q_dots, us = equally_spaced_in_place(points, i, shortest, dict_res, Ns[i])
        qs, q_dots, us = equally_spaced_in_time(i, shortest, dict_res, Ns[i])
        qs_reshaped = np.array(qs)
        q_dots_reshaped = np.array(q_dots)
        us_reshaped = np.array(us)
        end_effector_pose_func = mobile_robot.end_effector_pose_func()
        for k in range(Ns[i]): 
            if k == 0 and i == 0:
                X0_no_tn.extend([*qs_reshaped[:,k], *q_dots_reshaped[:,k], *us_reshaped[:,k]])
            else:
                X0_no_tn.extend([*qs_reshaped[:,k]+position_noise*np.random.random(qs_reshaped[:,k].shape), *q_dots_reshaped[:,k]+velocity_noise*np.random.random(q_dots_reshaped[:,k].shape), *us_reshaped[:,k]+acceleration_noise*np.random.random(us_reshaped[:,k].shape)])

            # X0_no_tn.extend([1]*(3*lambda_num_cols))
            if not is_sequential_guess:
                # append lambdas
                if k == Ns[i]-1:
                    X0_no_tn.extend([0]*(i+1) + [1]*(prediction_horizon-i-1))
                else:
                    X0_no_tn.extend([0]*(i) + [1]*(prediction_horizon-i))
                # append nus and mus
                # compute distance to all points
                norms_array = []
                mus_array = []
                for j in range(1,prediction_horizon+1):
                    if j==len(points):
                        point = points[0]
                    else:
                        point = points[j]
                    p = np.array([point[0], point[1]])
                    norm = np.linalg.norm(end_effector_pose_func([*qs_reshaped[:,k]]) - p)
                    if norm <= d_tol:
                        mus_array.append(1)
                        norms_array.append(norm**2)
                    else:
                        mus_array.append(0)
                        norms_array.append(d_tol**2)
                # print('point:', x[k], y[k])
                # print(cpc(mus_array, [x[k], y[k], 0, 0], norms_array))
                X0_no_tn.extend(mus_array)
                X0_no_tn.extend(norms_array)                                                    
    if is_sequential_guess:
        return X0_no_tn, ts, Ns
    return X0_no_tn, tf


def equally_spaced_in_time(i, shortest, dict_res, N=100):
    params = dict_res[shortest[i]][shortest[i+1]]
    if i ==0:
        t_values = np.linspace(0, params[0][6], N)
    else:
        t_values = np.linspace(0, params[0][6], N+1) # this way, there will not be two overlapping points
        t_values = t_values[1:]

    qs = []
    q_dots = []
    us = []
    for i in range(len(params)):
        qs.append([space_curve(t, params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8]) for t in t_values])
        q_dots.append([velocity_curve(t, params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8]) for t in t_values])
        us.append([acceleration_curve(t, params[i][2], params[i][5], params[i][7], params[i][8]) for t in t_values])
    return qs, q_dots, us

def equally_spaced_in_place(points, i, shortest, dict_res, N=100):
    qs_bad, q_dots, us = equally_spaced_in_time(i, shortest, dict_res, N)

    qs = []
    params = dict_res[shortest[i]][shortest[i+1]]
    for i in range(len(qs_bad)):
        starting_point = space_curve(0, params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8])
        ending_point = space_curve(params[i][6], params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8])
        qs.append(np.linspace(starting_point, ending_point, N))

    return qs, q_dots, us

def compute_tf(shortest, dict_res):
    tf = 0
    for i in range(len(shortest)-1):
        tf += dict_res[shortest[i]][shortest[i+1]][0][6]
    return tf

def compute_Ns(shortest, dict_res, N):
    Ns = []
    #
    tf = compute_tf(shortest, dict_res)
    ts = np.linspace(0, tf, N)
    t_pref = 0
    for i in range(len(shortest)-1):
        t = dict_res[shortest[i]][shortest[i+1]][0][6]
        # find how many points to sample from t_pref to t
        N_to_append = int(N*(t-t_pref)/tf)
        if i == len(shortest)-2:
            N_to_append = N - sum(Ns)
        Ns.append(N_to_append)
    # print('Ns:', Ns)
    return Ns, tf 