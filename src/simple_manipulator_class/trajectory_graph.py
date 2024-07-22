
def create_graph(self, waypoints_poses, end_effector_goals, sampling_rate=3):
    dict_graph = {}
    dict_graph[0] = [waypoints_poses[0]]
    for i in range(1,len(waypoints_poses)):
        dict_graph[i] = [waypoints_poses[i]]
    return dict_graph

        