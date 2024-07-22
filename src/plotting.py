import matplotlib.pyplot as plt
import numpy as np

from .point_mass_trajectory_optimization import space_curve, velocity_curve, acceleration_curve
from .decompose_casadi_solution import decompose_X

def plot_single_trajectory(params):
    ''' plot a single trajectory given the params.'''
    plt.figure(figsize=(12, 6))
    
    for i in range(len(params)):
        a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, tf_x, ts_x, ts_1_x = params[i]
        t_values = np.linspace(0, tf_x, 1000)
        x = [space_curve(t, a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        vx = [velocity_curve(t, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        plt.subplot(2, 2, 1)
        plt.plot(t_values, x, label='variable {}'.format(i))
        plt.title('Space Curves')
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.subplot(2, 2, 2)
        plt.plot(t_values, vx, label='v of variable {}'.format(i))
        plt.title('Velocity Curves')
        plt.xlabel('Time')
        plt.ylabel('Velocity')

    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_all_trajectories_and_shortest_path_2d(results, points, dict_res, shortest):
    '''plot all x-y trajectories for each scenario in the results list, the points to visit and the shortest path.'''
    plt.figure(figsize=(12, 6))

    for i, element in enumerate(results): 
        x_result, y_result = element
        a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, tf_x, ts_x, ts_1_x = x_result
        a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, tf_y, ts_y, ts_1_y = y_result

        t_values = np.linspace(0, tf_x, 1000)
        x = [space_curve(t, a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        y = [space_curve(t, a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, ts_y, ts_1_y) for t in t_values]
        
        plt.subplot(2, 2, 1)
        plt.plot(x, y, label='scenario {}'.format(i), color='grey', linewidth=0.5)
        plt.title('Space Curve')
        plt.xlabel('x')
        plt.ylabel('y')
    plt.scatter([point[0] for point in points], [point[1] for point in points], color='red')
    for i in range(len(shortest)-1):
        x_result, y_result = dict_res[shortest[i]][shortest[i+1]]
        a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, tf_x, ts_x, ts_1_x = x_result
        a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, tf_y, ts_y, ts_1_y = y_result

        t_values = np.linspace(0, tf_x, 1000)
        x = [space_curve(t, a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        y = [space_curve(t, a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, ts_y, ts_1_y) for t in t_values]
        print("starting velocity: ",(a1_x, a1_y))
        plt.subplot(2, 2, 1)
        plt.plot(x, y, label='scenario {}'.format(i), color='red', linewidth=0.7)
        plt.title('Space Curve')
        plt.xlabel('x')
        plt.ylabel('y')

    plt.tight_layout()
    plt.show()



def plot_velocities_in_cone(df, points):
    '''plot the initial velocities within the defined cone for each point in the points list.'''
    fig, ax = plt.subplots()
    print(df)
    # Define a colormap for different starting points
    colormap = plt.cm.get_cmap('viridis', len(df.keys()))
    for i, cone_velocities in df.items():
        x, y = points[i]
        print(cone_velocities)
        print()
        # Plot velocities as vectors with different colors
        vx = np.array([item[0] for item in cone_velocities])
        vy = np.array([item[1] for item in cone_velocities])
        ax.quiver([x] * len(cone_velocities), [y] * len(cone_velocities), vx, vy, color=colormap(i), scale=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Initial Velocities within Cone')
    plt.show()

def plot_multiple_trajectories_and_velocities(results):
    '''plot all x-y trajectories and their corresponding velocity curves for each scenario in the results list.'''
    plt.figure(figsize=(12, 6))

    for i, element in enumerate(results): 
        x_result, y_result = element
        a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, tf_x, ts_x, ts_1_x = x_result
        a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, tf_y, ts_y, ts_1_y = y_result

        t_values = np.linspace(0, tf_x, 1000)
        x = [space_curve(t, a0_x, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        y = [space_curve(t, a0_y, a1_y, a2_y, a3_y, a4_y, a5_y, ts_y, ts_1_y) for t in t_values]
        vx = [velocity_curve(t, a1_x, a2_x, a3_x, a4_x, a5_x, ts_x, ts_1_x) for t in t_values]
        vy = [velocity_curve(t, a1_y, a2_y, a3_y, a4_y, a5_y, ts_y, ts_1_y) for t in t_values]

        plt.subplot(2, 2, 1)
        plt.plot(x, y, label='scenario {}'.format(i))
        plt.title('Space Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(t_values, vx, label='vx of scenario {}'.format(i))
        # plt.plot(t_values, vy, label='vy of scenario {}'.format(i))
        plt.title('Velocity Curve')
        plt.xlabel('Time')
        plt.ylabel('Velocities')
        plt.legend()


    plt.tight_layout()
    plt.show()


def compare_trajectories_casadi_plot(casadi_results, points, dict_res, shortest, forward_kinematic, q_size=2, state_dim=[], labels=[]):
    # plot X
    number_of_plots = 1 + 2*q_size
    # define the subplots
    rows = int(np.ceil(np.sqrt(number_of_plots)))
    cols = int(np.ceil(number_of_plots / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)

    for i in range(len(casadi_results)):
        tf = float(casadi_results[i][0])
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])

        if labels:
            cur_label = labels[i]
        else:
            cur_label = f'{i}'
        # generate trajectory graph
        qs = [i for i in zip(*qs)]
        ee_list = []
        for i in range(len(qs)):
            ee_list.append(forward_kinematic(qs[i]))

        x = [float(point[0]) for point in ee_list]
        y = [float(point[1]) for point in ee_list]
        axes[0].plot(x, y, label=f'{cur_label} trajectory')

        t = np.linspace(0, tf, len(x))
        # plot velocities
        for j in range(q_size):
            v = [float(qs_dots[j][k]) for k in range(len(qs_dots[j]))]
            axes[j+1].plot(t, v, label=f'{cur_label} q{j} velocity')
        # plot accelerations
        for j in range(q_size):
            a = [float(us[j][k]) for k in range(len(us[j]))]
            axes[j+q_size+1].plot(t, a, label=f'{cur_label} q{j} acceleration')

    # plot fastest trajectory
    if dict_res!=None:
        tf = 0
        qs = []
        q_dots = []
        us = []
        times = []
        for i in range(len(shortest)-1):
            params = dict_res[shortest[i]][shortest[i+1]]
            t_values = np.linspace(0, params[0][6], 1000)
            for j in range(q_size):
                par = params[j]
                qs_section = [space_curve(t, par[0], par[1], par[2], par[3], par[4], par[5], par[7], par[8]) for t in t_values]
                q_dots_section = [velocity_curve(t, par[1], par[2], par[3], par[4], par[5], par[7], par[8]) for t in t_values]
                us_section = [acceleration_curve(t, par[2], par[5], par[7], par[8]) for t in t_values]
                if i == 0:
                    qs.append(qs_section)
                    q_dots.append(q_dots_section)
                    us.append(us_section)

                else:
                    qs[j]+=qs_section
                    q_dots[j]+=q_dots_section
                    us[j]+=us_section
            times += [t + tf for t in t_values]
            tf += params[0][6]
        #convert qs into ee
        qs = [i for i in zip(*qs)]
        ee_list = []
        for i in range(len(qs)):
            ee_list.append(forward_kinematic(qs[i]))
        xs = [float(point[0]) for point in ee_list]
        ys = [float(point[1]) for point in ee_list]
        axes[0].plot(xs, ys, label='Fastest trajectory, sampling algo', color='red')
        # add title
        axes[0].set_title('Trajectories')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        # plot points
        axes[0].scatter([point[0] for point in points], [point[1] for point in points], color='green', label='Points to visit')
        axes[0].legend()

        # plot velocities
        for j in range(q_size):
            v = [float(q_dots[j][k]) for k in range(len(q_dots[j]))]
            axes[j+1].plot(times, v, label=f'Fastest q{j} velocity', color='red')
            axes[j+1].set_title(f'Velocities of q{j}')
            axes[j+1].set_xlabel('Time')
            axes[j+1].set_ylabel('Velocity')
            axes[j+1].legend()
        # plot accelerations
        for j in range(q_size):
            a = [float(us[j][k]) for k in range(len(us[j]))]
            axes[j+q_size+1].plot(times, a, label=f'Fastest q{j} acceleration', color='red')
            axes[j+q_size+1].set_title(f'Accelerations of q{j}')
            axes[j+q_size+1].set_xlabel('Time')
            axes[j+q_size+1].set_ylabel('Acceleration')
            axes[j+q_size+1].legend()
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.show()