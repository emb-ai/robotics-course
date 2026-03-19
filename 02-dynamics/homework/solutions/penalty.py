"""Student solution: penalty contact forces. Task 3."""
from dataclasses import dataclass
from typing import List

from lib.phys.phys_objects import RigidBody, Particle
from lib.phys.contact_forces.handler import ContactHandlerBase
from lib.phys.collisions.collision_detector import Collision


@dataclass
class PenaltyMethod(ContactHandlerBase):
    k_s: float = 10000.0
    k_d: float = 20.0
    mu: float = 0.5
    k_drag: float = 10.0

    def step(self, collisions: List[Collision], t_0: float, t_1: float):
        # YOUR CODE HERE: for each collision, apply normal (spring-damper) and friction forces
        raise NotImplementedError("Implement penalty contact forces (normal + friction).")
