from .mobile_manipulator_class import MobileManipulator

def create_graph(self, waypoints_poses, end_effector_goals, sampling_rate=3):
    dict_graph = {}
    dict_graph[0] = [waypoints_poses[0]]
    for i in range(1,len(waypoints_poses)):
        possible_poses = self.displace_pose(waypoints_poses[i], end_effector_goals[i], x_increment=0.1, tolerance=0.05)
        # sample the possible poses
        sampled_poses = possible_poses[::sampling_rate]
        # Add the sampled pose to a graph
        dict_graph[i] = sampled_poses
    return dict_graph

        