from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from lib.basic_structs import Vec3, Pose
from lib.utils import skew_sym
from lib.phys.phys_objects import RigidBody


class Constraint(ABC):
    @abstractmethod
    def update(self, t_0, t_1):
        pass

    @property
    @abstractmethod
    def constrained_bodies(self):
        pass

    @property
    @abstractmethod
    def dim_size(self):
        pass

    @abstractmethod
    def get_C(self):
        pass

    @abstractmethod
    def get_J_updates(self):
        pass


    @abstractmethod
    def get_J_dot_V(self):
        pass

@dataclass
class BallAndSocketPoint(Constraint):
    '''
    Body point with coordinates r_f_local in the body frame
    is fixed in space. The body is free to rotate around this fixed point.

    r is the COM position of the body.
    At each timestep we transform r_f_local to r_f in the world frame.
    Initially, R := r_f at time t = 0 and is constant during the whole simulation.

    So the constraint uquation looks as follows:

    C(r) = r + r_f - R

    Args
    ----
    body: RigidBody
        Body under constraint.

    body_fixed_point_local: Vec3
        Body anchor point in body frame (r_f_local).
    '''
    body: RigidBody
    body_fixed_point_local: Vec3 # r_f_local

    def __post_init__(self,): # Calc R
        self.fixed_point_world = (self.body.pose * Pose(p=self.body_fixed_point_local)).p

    def update(self, t_0, t_1):
        '''
        find r_f from r_f_local
        '''
        self.body_fixed_point_world = self.body.pose.q.rotation_matrix @ self.body_fixed_point_local

    @property
    def constrained_bodies(self):
        return [self.body]

    @property
    def dim_size(self):
        return 3

    def get_C(self):
        '''
        C = r + r_f - R
        '''
        return self.body.pose.p + self.body_fixed_point_world - self.fixed_point_world 

    def get_J_updates(self):
        '''
        J = [ E | - [r_f]x ]
        '''
        r_f_star = skew_sym(self.body_fixed_point_world)
        return {
            self.body.name: np.hstack((np.eye(3), -r_f_star))
        }
    
    def get_J_dot_V(self):
        '''
        J'V = omega x (omega x r_f)
        '''
        return np.cross(self.body.angular_velocity, np.cross(self.body.angular_velocity, self.body_fixed_point_world))
