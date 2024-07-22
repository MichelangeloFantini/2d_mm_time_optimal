def decompose_X(X, state_dim, total_dim):
    x_new = X.full().flatten()
    qs = []
    q_dots = []
    us = []
    for i in range(state_dim):
        qs.append(x_new[i+1::total_dim])
        q_dots.append(x_new[i+1+state_dim::total_dim])
        us.append(x_new[i+1+2*state_dim::total_dim])
    return qs, q_dots, us
