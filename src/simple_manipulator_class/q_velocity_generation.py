import numpy as np

def generate_waypoints_q_dot(self, dict_graph, end_effector_velocity_dict, v_max, v_min):
    q_dot_dict = {}
    for i in range(len(dict_graph)):
        q_dot_dict[i] = {}
        for j in range(len(dict_graph[i])):
            q_dot_dict[i][j] = []
            for v0 in end_effector_velocity_dict[i]:
                current_pose = dict_graph[i][j]
                J = self.compute_jacobian(*current_pose)
                pseudo_inv_J = np.linalg.pinv(J)
                q_dot = pseudo_inv_J @ np.array([[v0[0]], [v0[1]]])
                # Check if the velocity is within the limits
                if self.check_velocity_in_bound(q_dot, v_max, v_min):
                    q_dot_dict[i][j].append((q_dot[0, 0], q_dot[1, 0]))
    return q_dot_dict

def check_velocity_in_bound(self, q_dot, v_max, v_min):
    for i in range(len(v_max)):
        if q_dot[i, 0] < v_min[i] or q_dot[i, 0] > v_max[i]:
            return False   
    return True
