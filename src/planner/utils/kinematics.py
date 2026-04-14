import copy
import numpy as np
from numpy.linalg import norm as norm
import random
import math
from typing import Tuple
from abc import abstractmethod
from scipy.spatial.transform import Slerp, Rotation as R
from .utils import SampleUnitNBall, SphereSampler
from scipy.spatial.distance import cdist

from scipy.spatial.transform import Rotation
def create_rotation_interpolation(R1: Rotation, R2: Rotation) -> Slerp:
    """ Create rotation interpolation function for two given rotations

    Args:
        R1 (Rotation): Rotation object
        R2 (Rotation): Rotation object
    
    Returns:
        rot_interp (Slerp): Spherical Linear Interpolation
    """
    rots = Rotation.concatenate([R1, R2])
    rot_interp = Slerp(np.array([0, 1]), rots)
    return rot_interp

def angular_difference(R1: Rotation, R2: Rotation) -> float:
    """ compute angular difference between two rotations

    Args:
        R1 (Rotation): Rotation object
        R2 (Rotation): Rotation object
    
    Returns:
        angle_diff (float): angular difference in radians
    """
    ### Calculate the angular difference between two rotations ###
    angular_diff = R1.inv() * R2
    
    ### Convert the angular difference to an angle ###
    angle_diff = angular_diff.magnitude()
    
    return angle_diff

def compute_camera_pose(A: np.ndarray, B: np.ndarray, up_dir: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """ compute camera pose given current location A and look-at location B.
    Using OpenGL (RUB) coordinate system. 
    up_dir is the up direction w.r.t world coorindate origin pose.

    Args:
        A (np.ndarray, [3])     : current location
        B (np.ndarray, [3])     : look-at location
        up_dir (np.ndarray, [3]): up direction in world coordinate
    
    Returns:
        M (np.ndarray, [3, 3]): rotation matrix
    """
    # viewing direction (backward)
    V = A - B

    ### FIXME: for edge case that target points in the same x,y position ###
    if V[0] == 0 and V[1] == 0:
        V[0] = 1e-6

    # right viewing direction
    R = np.cross(up_dir, V)

    # up viewing direction
    U = np.cross(V, R)

    # normalize
    V = V / np.linalg.norm(V)
    R = R / np.linalg.norm(R)
    U = U / np.linalg.norm(U)

    # construct pose matrix
    M = np.column_stack((R, U, V))  

    return M


class KinematicsBase(object):
    def __init__(self, dt:float):
        self.dt = dt

    @abstractmethod
    def transition(self, state, action):
        pass

class SE3Kinematics(KinematicsBase):
    def __init__(self, vmax, ang_max, v_ang_max, dt):
        super().__init__(dt)
        self.vmax = vmax
        self.ang_max = ang_max
        self.v_ang_max = v_ang_max

        self.rotate_roll_pitch = True

    # angular motion : rodrigues formula + postprocess for natural rotation
    def transition(self,
                   state: Tuple[np.ndarray, np.ndarray], 
                   action: Tuple[np.ndarray, np.ndarray],
                   dt=None,
                   use_space_frame=True) -> Tuple[np.ndarray, np.ndarray]:
        if dt is None:
            dt = self.dt
        pos, Rot = state
        if type(Rot) == np.ndarray and len(Rot)==4:
            Rot = R.from_quat(Rot)
        lin_vel, ang_vel = action
        
        new_pos = pos + lin_vel * dt
        
        wx, wy, wz = ang_vel
        S = np.array([[0, -wz, wy],
                      [wz, 0, -wx],
                      [-wy, wx, 0]]) # skew matrix
        
        ang_norm = np.linalg.norm(ang_vel * dt)
        ang_norm2 = ang_norm * ang_norm
        S2 = np.matmul(S,S)
        if use_space_frame:
            new_R = Rot.as_matrix() @ (np.eye(3)+np.sin(ang_norm)/(ang_norm+1e-10)*S*dt + (1-np.cos(ang_norm))/(ang_norm2+1e-10)*S2*dt*dt)
        else:
            new_R = (np.eye(3)+np.sin(ang_norm)/(ang_norm+1e-10)*S*dt + (1-np.cos(ang_norm))/(ang_norm2+1e-10)*S2*dt*dt)@Rot.as_matrix()

        new_R = R.from_matrix(new_R).as_matrix() # auto-normalize
        new_R = compute_camera_pose(np.zeros(3), -new_R[:,2])
        new_Rot = R.from_matrix(new_R)
        
        return new_pos, new_Rot
    
    def sample_linear_velocity(self, max_speed):
        sampled_direction = SphereSampler(np.zeros(3), 1.0)
        sampled_direction *= 1.0/np.linalg.norm(sampled_direction)
        linear_velocity = max_speed * sampled_direction
        return linear_velocity
    
    # angular velocity correspond to "angular_difference" function
    def relative_angular_velocity_v3(self, rotation_matrix, target_rotation, dt=None, ang_max=None):
        rot_diff = rotation_matrix.inv() * target_rotation
        ang_diff = rot_diff.magnitude()
        if ang_max is None:
            ang_max = self.ang_max
        if dt is None:
            dt = self.dt
        ang_vel_value = np.minimum(ang_diff, ang_max)

        if ang_diff < 1e-3:
            ang_vel = np.zeros(3)
            return ang_vel

        R = rot_diff.as_matrix()
        unit_dir = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        unit_dir = unit_dir/(2*np.sin(ang_diff))
        ang_vel = ang_vel_value * unit_dir / dt
        return ang_vel
