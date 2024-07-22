import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import casadi as ca

class PointMass:
    def __init__(self):
        self.name = 'PointMass'
        from . import motion_model
        setattr(PointMass, 'motion_model', motion_model.motion_model)
        setattr(PointMass, 'forward_kinematic', motion_model.forward_kinematic)
        setattr(PointMass, 'end_effector_pose_func', motion_model.end_effector_pose_func)
        from . import trajectory_computation
        setattr(PointMass, 'calculate_trajectory', trajectory_computation.calculate_trajectory)
