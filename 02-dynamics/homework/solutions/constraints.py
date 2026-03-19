"""Student solution: BallAndSocketJoint. Task 2.2. Re-exports BallAndSocketPoint from lib."""
import numpy as np
from dataclasses import dataclass

from lib.basic_structs import Vec3
from lib.phys.phys_objects import RigidBody
from lib.phys.constraints.constraints import BallAndSocketPoint, Constraint


@dataclass
class BallAndSocketJoint(Constraint):
    """Ball-and-socket joint: C = r_2 + r_f_2 - r_1 - r_f_1. Implement get_C and get_J_updates."""
    body_1: RigidBody
    body_2: RigidBody
    anchor_point_body_1_local: Vec3

    def __post_init__(self):
        anchor_point_world = (
            self.body_1.pose.p
            + self.body_1.pose.q.rotation_matrix @ self.anchor_point_body_1_local
        )
        anchor_2_point_world = anchor_point_world - self.body_2.pose.p
        self.anchor_point_body_2_local = (
            self.body_2.pose.q.rotation_matrix.T @ anchor_2_point_world
        )

    def update(self, t_0, t_1):
        self.body_1_fixed_point_world = (
            self.body_1.pose.q.rotation_matrix @ self.anchor_point_body_1_local
        )
        self.body_2_fixed_point_world = (
            self.body_2.pose.q.rotation_matrix @ self.anchor_point_body_2_local
        )

    @property
    def constrained_bodies(self):
        return [self.body_1, self.body_2]

    @property
    def dim_size(self):
        return 3

    def get_C(self):
        """C = r_2 + r_f_2 - r_1 - r_f_1."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement constraint C for ball-and-socket joint.")

    def get_J_updates(self):
        """Return {body_1.name: J_1, body_2.name: J_2} so dC/dt = J_1 V_1 + J_2 V_2."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement Jacobian updates for ball-and-socket joint.")

    def get_J_dot_V(self):
        """Not needed for impulse-based formulation; return zeros for concatenation."""
        return np.zeros(3)


__all__ = ["BallAndSocketPoint", "BallAndSocketJoint"]
