import casadi as ca

def generate_balls(self, params):
    ''' The goal of the functions is to generate balls around the mobile manipulator structure that can be then used for obstacle avoidance,
     given an initial configuaration.'''

    # Extract the parameters
    x, theta1, theta2 = params
    x0, y0, x1, y1, x2, y2, x3, y3 = self.compute_link_positions(x, theta1, theta2)
    # a ball per link and two balls per base
    balls = []
    # Base balls
    balls.append([x0-self.base_width/2, y0])
    balls.append([x0+self.base_width/2, y0])
    # Link balls
    balls.append([x1, y1])
    balls.append([x2, y2])
    balls.append([x3, y3])
    return balls

def generate_balls_constraints(self, params):
    func = self.link_position_function()
    res = func(params)
    # a ball per link and two balls per base
    balls = []
    # Base balls
    x0, y0 = res[0][0], res[0][1]
    balls.append(ca.vertcat(x0-self.base_width/2, y0))
    balls.append(ca.vertcat(x0+self.base_width/2, y0))
    # Link balls
    x1, y1 = res[1][0], res[1][1]
    x2, y2 = res[2][0], res[2][1]
    x3, y3 = res[3][0], res[3][1]
    balls.append(ca.vertcat(x1, y1))
    balls.append(ca.vertcat(x2, y2))
    balls.append(ca.vertcat(x3, y3))
    return balls