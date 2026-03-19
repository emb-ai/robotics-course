from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, List, Union
import numpy as np

from lib.basic_structs import Pose, Vec3
from lib.phys.phys_objects import RigidBody, Particle

@dataclass
class ForceBase(ABC):
    @abstractmethod
    def apply_force(self, bodies: Dict[str, RigidBody]):
        pass

@dataclass
class GravityForce(ForceBase):
    g_vector: Vec3 = Vec3(0, -9.81, 0)

    def apply_force(self, bodies: List[RigidBody]):
        for body in bodies:
            if body.inv_mass > 0:
                body.force_accumulator += self.g_vector / body.inv_mass

@dataclass
class HookeForce(ForceBase):
    body_a: Union[Particle, RigidBody]
    body_b: Union[Particle, RigidBody]
    ks: float
    kd: float
    application_point_a: Vec3 = Vec3(0, 0, 0)
    application_point_b: Vec3 = Vec3(0, 0, 0)
    rest_length: float = 1.0

    def calc_force(self):
        com_a_pose = Pose(self.body_a.pos) if isinstance(self.body_a, Particle) else self.body_a.pose
        com_b_pose = Pose(self.body_b.pos) if isinstance(self.body_b, Particle) else self.body_b.pose
    
        a = (com_a_pose * Pose(self.application_point_a)).p
        b = (com_b_pose * Pose(self.application_point_b)).p
        l = a - b

        r_a = a - com_a_pose.p
        r_b = b - com_b_pose.p

        a_vel = self.body_a.body_velocity
        if isinstance(self.body_a, RigidBody):
            a_vel += np.cross(self.body_a.angular_velocity, r_a)
            
        b_vel = self.body_b.body_velocity 
        if isinstance(self.body_b, RigidBody):
            b_vel += np.cross(self.body_b.angular_velocity, r_b)
            
        l_dot = a_vel - b_vel

        delta_l = np.linalg.norm(l) - self.rest_length
        delta_l_dot = np.dot(l_dot, l) / (np.linalg.norm(l) + 1e-3)

        module = self.ks * delta_l + self.kd * delta_l_dot
        return -module * l / (np.linalg.norm(l) + 1e-3)

    def apply_force(self, bodies):
        hooke_force = self.calc_force()

        self.body_a.force_accumulator += hooke_force
        self.body_b.force_accumulator += -hooke_force

        if isinstance(self.body_a, RigidBody):
            r_a = (self.body_a.pose * Pose(self.application_point_a)).p - self.body_a.pose.p
            torque_a = np.cross(r_a, hooke_force)
            self.body_a.torque_accumulator += torque_a

        if isinstance(self.body_b, RigidBody):
            r_b = (self.body_b.pose * Pose(self.application_point_b)).p - self.body_b.pose.p
            torque_b = np.cross(r_b, -hooke_force)
            self.body_b.torque_accumulator += torque_b
