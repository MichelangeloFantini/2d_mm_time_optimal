import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation

from ..point_mass_trajectory_optimization import space_curve, velocity_curve

def plot_static(self, theta1, theta2, goal):
        x1, y1, x2, y2, x3, y3 = self.compute_link_positions(theta1, theta2)

        # Plotting
        fig, ax = plt.subplots()

        # Plotting mobile base (rectangle)
        base_rect = Rectangle((self.base_x - self.base_length / 2, self.base_y - self.base_width / 2), self.base_length, self.base_width, color='gray', alpha=0.5)
        ax.add_patch(base_rect)


        # Plotting manipulator arms (rectangles)
        arm1 = Rectangle((x1, y1), self.arm_length1, self.arm_width1, angle=np.degrees(theta1), color='purple')
        arm2 = Rectangle((x2, y2), self.arm_length2, self.arm_width2, angle=np.degrees(theta1 + theta2), color='green')
        ax.add_patch(arm1)
        ax.add_patch(arm2)

        # Plotting end-effector
        ax.plot(x3, y3, 'ko')  # End-effector
        ax.plot(goal[0], goal[1], 'ro')  # Goal
        plt.xlim(x2-5, x2+5)
        plt.ylim(0, y2+2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Mobile Manipulator with Base and Arms')
        plt.grid(True)
        plt.show()
    
def plot_dynamic(self, start_point, dict_res, shortest, points):
    results, times = self.process_results(dict_res, shortest)
    def update(frame):
        # Update joint angles
        for i in range(len(times)):
            if frame*self.interval*(0.001) < times[i]:
                a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, tf_theta1, ts_theta1, ts_1_theta1 = results[i][0]
                a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, tf_theta2, ts_theta2, ts_1_theta2 = results[i][1]
                if i == 0:
                    t_offset = 0
                else:
                    t_offset = times[i-1]
                break
        theta1 = space_curve(frame*self.interval*(0.001) - t_offset, a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1)
        theta2 = space_curve(frame*self.interval*(0.001) - t_offset, a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2)
        x1, y1, x2, y2, x3, y3 = self.compute_link_positions(theta1, theta2)

        # Update manipulator configuration
        arm1.set_xy((x1, y1))
        arm1.angle = np.degrees(theta1)
        arm2.set_xy((x2, y2))
        arm2.angle = np.degrees(theta1 + theta2)
        plt.xlim(x2-5, x2+5)
        plt.ylim(0, y2+2)
        return arm1, arm2

    x1, y1, x2, y2, x3, y3 = self.compute_link_positions(start_point[0], start_point[1])
    theta1 = start_point[0]
    theta2 = start_point[1]

    # Calculate the total time taken
    t_final = times[-1]

    # Plotting
    fig, ax = plt.subplots()

    # Plotting mobile base (rectangle)
    base_rect = Rectangle((self.base_x - self.base_length / 2, self.base_y - self.base_width / 2), self.base_length, self.base_width, color='gray', alpha=0.5)
    ax.add_patch(base_rect)

    # Plotting manipulator arms (rectangles)
    arm1 = Rectangle((x1, y1), self.arm_length1, self.arm_width1, angle=np.degrees(theta1), color='purple')
    arm2 = Rectangle((x2, y2), self.arm_length2, self.arm_width2, angle=np.degrees(theta1 + theta2), color='green')
    ax.add_patch(arm1)
    ax.add_patch(arm2)

    # Set plot limits and aspect ratio
    plt.xlim(x2-5, x2+5)
    plt.ylim(0, y2+2)
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    ax.scatter(x, y, color='red')

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, int(t_final/(self.interval*0.001))), interval=self.interval, blit=True)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Mobile Manipulator with Base and Arms')
    plt.grid(True)
    plt.show()

def plot_dynamic_params(self, params, interval=50, points=None):
    def update(frame):
        # Update joint angles
        theta1, theta2 = params[frame]
        x1, y1, x2, y2, x3, y3 = self.compute_link_positions(theta1, theta2)

        # Update manipulator configuration
        arm1.set_xy((x1, y1))
        arm1.angle = np.degrees(theta1)
        arm2.set_xy((x2, y2))
        arm2.angle = np.degrees(theta1 + theta2)
        plt.xlim(x2-5, x2+5)
        plt.ylim(0, y2+2)
        return arm1, arm2

    x1, y1, x2, y2, x3, y3 = self.compute_link_positions(params[0][0], params[0][1])
    theta1 = params[0][0]
    theta2 = params[0][1]

    # Plotting
    fig, ax = plt.subplots()

    # Plotting mobile base (rectangle)
    base_rect = Rectangle((self.base_x - self.base_length / 2, self.base_y - self.base_width / 2), self.base_length, self.base_width, color='gray', alpha=0.5)
    ax.add_patch(base_rect)

    # Plotting manipulator arms (rectangles)
    arm1 = Rectangle((x1, y1), self.arm_length1, self.arm_width1, angle=np.degrees(theta1), color='purple')
    arm2 = Rectangle((x2, y2), self.arm_length2, self.arm_width2, angle=np.degrees(theta1 + theta2), color='green')
    ax.add_patch(arm1)
    ax.add_patch(arm2)

    # Set plot limits and aspect ratio
    plt.xlim(x2-5, x2+5)
    plt.ylim(0, y2+2)
    plt.gca().set_aspect('equal', adjustable='box')

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(params)), interval=interval*1000, blit=True)
    if points is not None:
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        ax.scatter(x, y, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Mobile Manipulator with Base and Arms')
    plt.grid(True)
    plt.show()

def plot_end_effector(self, result, dict_res, shortest, points):
    for i in result:
        a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, tf_theta1, ts_theta1, ts_1_theta1 = i[0]
        a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, tf_theta2, ts_theta2, ts_1_theta2 = i[1]
        t_values = np.linspace(0, tf_x, 1000)
        theta1 = [space_curve(t, a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1) for t in t_values]
        theta2 = [space_curve(t, a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2) for t in t_values]
        x3_values = []
        y3_values = []
        for j in range(len(t_values)):
            x1, y1, x2, y2, x3, y3 = self.compute_link_positions( theta1[j], theta2[j])
            x3_values.append(x3)
            y3_values.append(y3)
        plt.plot(x3_values, y3_values, 'grey')
    for i in range(len(shortest)-1):
        theta1_res, theta2_res = dict_res[shortest[i]][shortest[i+1]]
        a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, tf_theta1, ts_theta1, ts_1_theta1 = theta1_res
        a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, tf_theta2, ts_theta2, ts_1_theta2 = theta2_res
        t_values = np.linspace(0, tf_x, 1000)
        x = [space_curve(t, a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        theta1 = [space_curve(t, a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1) for t in t_values]
        theta2 = [space_curve(t, a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2) for t in t_values]
        # Compute end effector position
        x3_values = []
        y3_values = []
        for j in range(len(t_values)):
            x1, y1, x2, y2, x3, y3 = self.compute_link_positions(theta1[j], theta2[j])
            x3_values.append(x3)
            y3_values.append(y3)
        plt.plot(x3_values, y3_values, 'red')
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('End Effector Trajectory')
    plt.grid(True)
    plt.show()

def plot_end_effector_best_trajectory_and_velocity(self, result, dict_res, shortest, points):
    x3_values = []
    y3_values = []
    v_theta1 = []
    v_theta2 = []
    ts = []
    tf = 0
    for i in range(len(shortest)-1):
        theta1_res, theta2_res = dict_res[shortest[i]][shortest[i+1]]
        a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, tf_theta1, ts_theta1, ts_1_theta1 = theta1_res
        a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, tf_theta2, ts_theta2, ts_1_theta2 = theta2_res
        t_values = np.linspace(0, tf_theta1, 1000)

        theta1 = [space_curve(t, a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1) for t in t_values]
        theta2 = [space_curve(t, a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2) for t in t_values]
        v_theta1_temp = [velocity_curve(t, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1) for t in t_values]
        v_theta2_temp = [velocity_curve(t, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2) for t in t_values]
        v_theta1.extend(v_theta1_temp)
        v_theta2.extend(v_theta2_temp)
        # Compute end effector position
        x3_values_temp = []
        y3_values_temp = []
        for j in range(len(t_values)):
            x1, y1, x2, y2, x3, y3 = self.compute_link_positions( theta1[j], theta2[j])
            x3_values_temp.append(x3)
            y3_values_temp.append(y3)
        x3_values.extend(x3_values_temp)
        y3_values.extend(y3_values_temp)
        ts.extend(t_values+tf)
        tf += tf_theta1
    plt.subplot(2, 1, 1)
    plt.plot(x3_values, y3_values, 'red', label='End effector trajectory')
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('End Effector Trajectory')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(ts, v_theta1, label='Theta1 velocity')
    plt.plot(ts, v_theta2, label='Theta2 velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocities over time')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_end_effector_best(self, result, dict_res, shortest, points):
    for i in range(len(shortest)-1):
        theta1_res, theta2_res = dict_res[shortest[i]][shortest[i+1]]
        a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, tf_theta1, ts_theta1, ts_1_theta1 = theta1_res
        a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, tf_theta2, ts_theta2, ts_1_theta2 = theta2_res
        t_values = np.linspace(0, tf_x, 1000)
        theta1 = [space_curve(t, a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1) for t in t_values]
        theta2 = [space_curve(t, a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2) for t in t_values]
        # Compute end effector position
        x3_values = []
        y3_values = []
        for j in range(len(t_values)):
            x1, y1, x2, y2, x3, y3 = self.compute_link_positions(theta1[j], theta2[j])
            x3_values.append(x3)
            y3_values.append(y3)
        plt.plot(x3_values, y3_values, 'red')
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('End Effector Trajectory')
    plt.grid(True)
    plt.show()

def plot_end_effector_and_velocities(self, result, dict_res, shortest, points):
    for num, i in enumerate(result):
        a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, tf_theta1, ts_theta1, ts_1_theta1 = i[0]
        a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, tf_theta2, ts_theta2, ts_1_theta2 = i[1]
        t_values = np.linspace(0, tf_x, 1000)
        theta1 = [space_curve(t, a0_theta1, a1_theta1, a2_theta1, a3_theta1, a4_theta1, a5_theta1, ts_theta1, ts_1_theta1) for t in t_values]
        theta2 = [space_curve(t, a0_theta2, a1_theta2, a2_theta2, a3_theta2, a4_theta2, a5_theta2, ts_theta2, ts_1_theta2) for t in t_values]
        x3_values = []
        y3_values = []

        for j in range(len(t_values)):
            x1, y1, x2, y2, x3, y3 = self.compute_link_positions(theta1[j], theta2[j])
            x3_values.append(x3)
            y3_values.append(y3)
        plt.subplot(2, 2, 1)
        plt.plot(x3_values, y3_values, label='scenario {}'.format(num))
        plt.title('End Effector Trajectory')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.legend()


        plt.subplot(2, 2, 2)
        plt.plot(t_values, vx, label='v of scenario {}'.format(num))
        plt.title('base velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.legend()

    plt.tight_layout()
    plt.show()