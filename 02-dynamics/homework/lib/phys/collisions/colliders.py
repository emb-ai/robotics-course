from dataclasses import dataclass
import fcl

from lib.basic_structs import Pose, Vec3

@dataclass
class ColliderBase(object):
    fcl_collider: fcl.CollisionObject = None

    def update_collider(self, new_pos: Vec3 = None, new_pose: Pose = None):
        t = fcl.Transform()
        if new_pos is not None:
            t = fcl.Transform(new_pos)
        if new_pose is not None:
            t = fcl.Transform(new_pose.q.q, new_pose.p)
        self.fcl_collider.setTransform(t)

@dataclass
class BoxCollider(ColliderBase):
    @classmethod
    def create(cls, extents, pose: Pose = Pose()):
        collider = fcl.CollisionObject(fcl.Box(*extents), fcl.Transform(pose.q.q, pose.p))
        return cls(fcl_collider=collider)


@dataclass
class SphereCollider(ColliderBase):
    @classmethod
    def create(cls, radius, pos: Vec3 = Vec3(0, 0, 0)):
        collider = fcl.CollisionObject(fcl.Sphere(radius), fcl.Transform(pos))
        return cls(fcl_collider=collider)