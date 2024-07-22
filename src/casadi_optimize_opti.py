import casadi as ca


def optimize_opti(points, v_max, v_min, a_max, a_min, prediction_horizon, X0, t_bound, motion_model, forward_kinematic, N=100, d_tol=0.01, constrain_final_point=False):

    opti = ca.Opti()

    X = opti.variable(3*len(v_max) + 3*(prediction_horizon), N)

    state_dim = len(v_max)
    x_dyn_num_cols = 2*state_dim
    control_num_cols = state_dim
    lambda_num_cols = prediction_horizon #assume starting point is not waypoint
    total_elements = 3*len(v_max) + 3*(prediction_horizon)


    vel_start_index = state_dim 
    u_start_index = x_dyn_num_cols 
    lambda_start_index =u_start_index + control_num_cols  
    mu_start_index = lambda_start_index + prediction_horizon  
    nu_start_index = mu_start_index + prediction_horizon 

    Xs = X[:u_start_index, :]
    Us = X[u_start_index:lambda_start_index, :]  
    Lambdas = X[lambda_start_index:mu_start_index, :]
    Mus = X[mu_start_index:nu_start_index, :]
    Nus = X[nu_start_index:, :]

    tn = opti.variable()

    # Goal: minimize time
    opti.minimize(tn)
    # Bounds for tn
    opti.subject_to(tn <= t_bound)
    opti.subject_to(tn >= 0)
    # Set bounds for X0
    opti.subject_to(Xs[:x_dyn_num_cols,0] - X0[1:x_dyn_num_cols+1].T == 0)
    if constrain_final_point:
        opti.subject_to(forward_kinematic(Xs[:state_dim,-1]) - [points[prediction_horizon][0], points[prediction_horizon][1]] ==0)
    # Bounds for U
    for k in range(N):
        for i in range(state_dim):
            opti.subject_to(Us[i,k] <= a_max[i])
            opti.subject_to(Us[i,k] >= a_min[i])
    # Bounds for V
    for k in range(N):
        for i in range(state_dim):
            opti.subject_to(Xs[vel_start_index+i,k] <= v_max[i])
            opti.subject_to(Xs[vel_start_index+i,k] >= v_min[i])
    # Bounds for Lambda
    opti.subject_to(Lambdas[:,0] == 1)
    opti.subject_to(Lambdas[:,-1] == 0)
    # Bounds for Mu
    for i in range(N):
        opti.subject_to(Mus[:,i] <= 1)

        opti.subject_to(Mus[:,i] >= 0)
    # Bounds for Nu
    for i in range(N):
        opti.subject_to(Nus[:,i] <= d_tol**2)
        opti.subject_to(Nus[:,i] >= 0)

    # Motion model constraints
    x_dyn = ca.MX.sym('x_dyn', 2*state_dim) # assume x_dyn looks like [x0, x1, v0, v1], column vector
    control = ca.MX.sym('control', state_dim) # assume control looks like [u0, u1], column vector

    dt = tn/(N)
    for i in range(N-1):
        opti.subject_to(Xs[:,i+1] - Xs[:,i] - dt*motion_model(Xs[:,i], Us[:,i])==0)

    # Lambda constraints
    for i in range(N-1):
        opti.subject_to(Lambdas[:,i+1] - Lambdas[:,i] + Mus[:,i]==0)

    # CPC constraints
    for i in range(N):
        for j in range(1,prediction_horizon+1):
            if j==len(points):
                point = points[0]
            else:
                point = points[j]
            p = ca.DM([point[0], point[1]])
            norm = ca.norm_2(forward_kinematic(Xs[:state_dim,i]) - p)
        
            opti.subject_to((Mus[j-1,i] * ((norm**2) - Nus[j-1,i])) <= 0.00001)
            opti.subject_to((Mus[j-1,i] * ((norm**2) - Nus[j-1,i])) >= 0)


    # Lex order
    for i in range(N):
        for j in range(prediction_horizon-1):
            opti.subject_to(Lambdas[j,i] - Lambdas[j+1,i]<=0)

    opti.set_initial(tn, X0[0])
    # print('X0:', X0[1:].reshape((N, 3*len(v_max) + 3*(prediction_horizon))))
    # opti.set_initial(X, X0[1:].reshape((N, 3*len(v_max) + 3*(prediction_horizon))).T)
    Xs_guess = X0[1::total_elements]
    for i in range(1,total_elements):
        Xs_guess = ca.vertcat(Xs_guess, X0[1+i::total_elements])

    opti.set_initial(X, Xs_guess)
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve()

        X_sol = sol.value(X)
        result = ca.horzcat(sol.value(tn), sol.value(X).T.reshape((1,-1)))
    except Exception as e:
        print(e)
        result = ca.horzcat(opti.debug.value(tn), opti.debug.value(X).T.reshape((1,-1)))

    return result, total_elements

    