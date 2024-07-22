import casadi as ca

def robot_motion_model(self):
    theta1 = ca.MX.sym('theta1')
    theta2 = ca.MX.sym('theta2')

    t1_dot = ca.MX.sym('t1_dot')
    t2_dot = ca.MX.sym('t2_dot')

    q = ca.vertcat(theta1, theta2)
    q_dot = ca.vertcat(t1_dot, t2_dot)
    u = ca.MX.sym('u', 2)
    inputs = ca.vertcat(q, q_dot)

    l1 = self.arm_length1
    I2 = (1/3)*self.m2*(self.arm_length1**2)
    I3 = (1/3)*self.m3*(self.arm_length2**2)
    c2 = 0.5*self.m2*self.arm_length1
    c3 = 0.5*self.m3*self.arm_length2
    g = 9.81
    l_c_m_2 = 0.5*self.arm_length1
    l_c_m_3 = 0.5*self.arm_length2

    M = ca.vertcat(
        ca.horzcat((l1**2)*self.m3 + (2*c3*ca.cos(theta2)*l1)+ I3 + I2, (c3*ca.cos(theta2)*l1) + I3),
        ca.horzcat( (l1*ca.cos(theta2)*c3) + I3, I3)
    )

    fg = ca.vertcat(self.m2*g*ca.cos(theta1)*l_c_m_2 + self.m3*g*(ca.cos(theta1+theta2)*l_c_m_3 + l1*ca.cos(theta1)), 
                    self.m3*g*(ca.cos(theta1+theta2)*l_c_m_3))


    # M_dot = ca.vertcat(
    #     ca.horzcat(2*t2_dot*l1*c3*ca.sin(theta2), -c3*l1*t2_dot*ca.sin(theta2)),
    #     ca.horzcat(-c3*l1*t2_dot*ca.sin(theta2), 0)
    # )

    # d_M_d_t1 = ca.jacobian(M, theta1).reshape((2, 2))
    # d_M_d_t2 = ca.jacobian(M, theta2).reshape((2, 2))


    # h = M_dot@q_dot + ca.vertcat(q_dot.T @ d_M_d_t1 @ q_dot, 
    #                              q_dot.T @ d_M_d_t2 @ q_dot)


    h = ca.vertcat(
        (-c3*l1*(2*t1_dot + t2_dot)*t2_dot*ca.sin(theta2)), c3*l1*ca.sin(theta2)*(t1_dot**2)
    )

    n_zeros = ca.DM.zeros(2, 2)
    n_ones = ca.DM.eye(2)
    matrix = ca.vertcat(
        ca.horzcat(n_zeros, n_ones),
        ca.DM.zeros(2, 4)
    )
    rhs_motion = matrix @ inputs + ca.vertcat(ca.DM.zeros(2), ca.inv(M)@(u -h -fg ))
    # rhs_motion = matrix @ inputs + ca.vertcat(ca.DM.zeros(2), u)

    motion_model = ca.Function('motion_model',[inputs, u], [rhs_motion])
    # print('done')
    return motion_model

def new_motion_model(self):
    theta1 = ca.MX.sym('theta1')
    theta2 = ca.MX.sym('theta2')

    t1_dot = ca.MX.sym('t1_dot')
    t2_dot = ca.MX.sym('t2_dot')

    q = ca.vertcat(theta1, theta2)
    q_dot = ca.vertcat(t1_dot, t2_dot)
    u = ca.MX.sym('u', 2)
    inputs = ca.vertcat(q, q_dot)

    m1 = self.m2
    m2 = self.m3

    l1 = self.arm_length1
    l2 = self.arm_length2
    I1 = (1/12)*self.m2*(self.arm_length1**2)
    I2 = (1/12)*self.m3*(self.arm_length2**2)
    r1 = 0.5*self.arm_length1
    r2 = 0.5*self.arm_length2
    g = 9.81


    M = ca.vertcat(
        ca.horzcat(m1 *(r1**2) + m2*((l1**2) + (r2**2) + 2*l1*(r2**2) + 2*l1*r2*ca.cos(theta2)) + I1 + I2, m2*((r2**2)+ l1*r2*ca.cos(theta2)) + I2),
        ca.horzcat(m2*((r2**2)+ l1*r2*ca.cos(theta2)) + I2, m2*(r2**2)+ I2)
        
    )

    h = -m2*l1*r2*ca.sin(theta2)

    coriolis_terms = ca.vertcat(
        h*t1_dot*t2_dot + h*t1_dot*t2_dot + h*(t2_dot**2),
        -h*(t1_dot**2)
    )

    gravity_terms = ca.vertcat(
        (m1*r1 + m2*r2)*g*ca.cos(theta1) + m2*r2*g*ca.cos(theta1 + theta2),
        m2*r2*ca.cos(theta1 + theta2)*g
    )

    n_zeros = ca.DM.zeros(2, 2)
    n_ones = ca.DM.eye(2)
    matrix = ca.vertcat(
        ca.horzcat(n_zeros, n_ones),
        ca.DM.zeros(2, 4)
    )
    rhs_motion = matrix @ inputs + ca.vertcat(ca.DM.zeros(2), ca.inv(M)@(u -coriolis_terms -gravity_terms ))

    motion_model = ca.Function('motion_model',[inputs, u], [rhs_motion])
    return motion_model