from dataclasses import dataclass, field
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
import fcl

from lib.phys.phys_objects import RigidBody, Particle
from lib.basic_structs import Vec3

@dataclass
class Collision:
    obj_a: Union[RigidBody, Particle]
    obj_b: Union[RigidBody, Particle]

    point_a: Vec3
    point_b: Vec3
    normal: Vec3
    depth: float


@dataclass
class CollisionDetectorBase(ABC):
    bodies: List[RigidBody] = field(default_factory=lambda: [])
    allowed_collisions: List[Tuple[str, str]] = field(default_factory=lambda: [])

    def __post_init__(self):
        for body in self.bodies:
            if body.collider is None:
                raise ValueError(f"Object {body.name} has no collider!")

    @abstractmethod
    def get_collisions(self):
        pass

    def add_object(self, new_obj: RigidBody):
        self.bodies.append(new_obj)

    def update_colliders(self):
        for body in self.bodies:
            if body.collider is None:
                continue
            if isinstance(body, RigidBody):
                body.collider.update_collider(new_pose=body.pose)
            else:
                body.collider.update_collider(new_pos=body.pos)


@dataclass
class FCLCollisionDetector(CollisionDetectorBase):
    def get_collisions(self):
        self.update_colliders()

        req = fcl.CollisionRequest(enable_contact=True, num_max_contacts=10,)

        collisions = []
        for body_a in self.bodies:
            if body_a.collider is None:
                continue
            for body_b in self.bodies:
                if body_b.collider is None:
                    continue
                if body_a.name == body_b.name: 
                    continue
                if (body_a.name, body_b.name) in self.allowed_collisions:
                    continue

                res = fcl.CollisionResult()
                _ = fcl.collide(body_a.collider.fcl_collider, body_b.collider.fcl_collider, req, res)
                for contact in res.contacts:
                    p1 = body_a.pose.p if isinstance(body_a, RigidBody) else body_a.pos
                    p2 = body_a.pose.p if isinstance(body_a, RigidBody) else body_a.pos
                    point_a = Vec3(contact.pos) - p1
                    point_b = Vec3(contact.pos) - p2
                    collisions.append(Collision(
                        obj_a = body_a,
                        obj_b = body_b,
                        point_a = point_a,
                        point_b = point_b,
                        normal = contact.normal,
                        depth = contact.penetration_depth
                    ))
        return collisions




