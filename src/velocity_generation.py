import numpy as np

def compute_velocities_in_cone_2d(points, module_min, module_max, vertex_angle_deg=30, magnitude_step=2, angle_step=20, loop=False, prediction_horizon=None):
    df = {}
    if not prediction_horizon:
        prediction_horizon = len(points)
    points_to_inspect = points[0:prediction_horizon+1]
    for i, point in enumerate(points_to_inspect):
        x, y = point
        # Convert vertex angle from degrees to radians
        vertex_angle_rad = np.deg2rad(vertex_angle_deg)

        # Generate all possible combinations of magnitude and angle
        magnitudes = np.linspace(module_min, module_max, magnitude_step)

        # Calculate the centerline vector for the current point
        if i < len(points_to_inspect) - 1:
            next_point = points_to_inspect[i + 1]
        else:
            if loop:
                next_point = points_to_inspect[0]  # Use initial point for the last point
            else:
                # next point is on the line between the last two points
                next_point = (2 * x - points_to_inspect[i - 1][0], 2 * y - points_to_inspect[i - 1][1])
        centerline_vector = np.array([next_point[0] - x, next_point[1] - y])

        # Filter velocities lying within the cone centered on the centerline vector
        cone_velocities = []
        for magnitude in magnitudes:
            # Calculate the angle of the velocity relative to the centerline vector
            centerline = np.arctan2(centerline_vector[1], centerline_vector[0])
            angle = centerline + np.linspace(-vertex_angle_rad, vertex_angle_rad, angle_step)
            vx = magnitude * np.cos(angle)
            vy = magnitude * np.sin(angle)
            if magnitude == 0 or i==0:
                vx = [0]
                vy = [0]
            
            #express velocities as points with x and y components
            combined_list = [(vx[i], vy[i]) for i in range(len(vx))]
            cone_velocities.extend(combined_list)
            # cone_velocities.append((vx, vy))
            if i==0:
                break   
        df[i] = cone_velocities
    return df