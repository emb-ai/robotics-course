from dataclasses import dataclass
from typing import Union
import vpython as vp
import numpy as np

from lib.phys.phys_objects import RigidBody, Particle
from lib.basic_structs import Vec3, Pose
from lib.phys.collisions.colliders import BoxCollider, SphereCollider

@dataclass
class Actor:
    name: str
    phys_obj: Union[RigidBody, Particle]
    visual: vp.baseObj
    visual_current_pose: Pose = None

    def __post_init__(self):
        self.update_visuals()

    def update_visuals(self):
        if isinstance(self.phys_obj, Particle):
            # particle
            self.visual.pos = vp.vector(*self.phys_obj.pos.tolist())
        else:
            # rigid body
            self.visual.pos = vp.vector(*self.phys_obj.pose.p.tolist())
    
            ox = self.phys_obj.pose.matrix[:3, 0]
            oy = self.phys_obj.pose.matrix[:3, 1]
    
            self.visual.axis = vp.vector(*ox.tolist()) * self.visual.length
            self.visual.up = vp.vector(*oy.tolist()) * self.visual.height


@dataclass
class VisualLinkage:
    name: str
    visual: vp.baseObj
    entity_1: Actor
    entity_2: Actor
    joint_1: Pose = Pose()
    joint_2: Pose = Pose()

    def update_visuals(self):
        if isinstance(self.entity_1.phys_obj, RigidBody):
            start = vp.vector(*(self.entity_1.phys_obj.pose * self.joint_1).p.tolist())
        else:
            start = vp.vector(*(Pose.from_pq(p=self.entity_1.phys_obj.pos) * self.joint_1).p.tolist())
        if isinstance(self.entity_2.phys_obj, RigidBody):
            end = vp.vector(*(self.entity_2.phys_obj.pose * self.joint_2).p.tolist())
        else:
            end = vp.vector(*(Pose.from_pq(p=self.entity_2.phys_obj.pos) * self.joint_2).p.tolist())
        axis = end - start

        self.visual.pos = start
        self.visual.axis = axis

@dataclass
class VisPhysObjVectorVar:
    visual: vp.baseObj
    entity: Actor
    phys_obj_variable_name: str
    local_transform: Pose = Pose()
    scale: float = 1.0
    def update_visuals(self):
        vec = self.entity.phys_obj.__getattribute__(self.phys_obj_variable_name)
        assert isinstance(vec, Vec3)

        start = vp.vector(*(self.entity.phys_obj.pose * self.local_transform).p.tolist())
        axis = vp.vector(*vec.tolist()) * self.scale

        self.visual.pos = start
        self.visual.axis = axis


def get_box_inertia(mass: int, extents: tuple):
    x, y, z = extents
    return mass / 12 * np.array(
        [
            [y ** 2 + z ** 2, 0, 0],
            [0, x ** 2 + z ** 2, 0],
            [0, 0, x ** 2 + y ** 2]
        ], dtype=np.float32
    )

def make_box_entity(
        name,
        extents=(1, 1, 1),
        color = vp.color.green,
        opacity=1.0,
        pos = [0, 0, 0],
        lin_momentum = [0, 0, 0],
        ang_momentum = [0, 0, 0],
        mass = 1.0
):
    body_inertia_tensor = get_box_inertia(mass, extents)
    return Actor(
        name = name,
        phys_obj=RigidBody(
            name = name,
            inv_mass = 1 / mass,
            body_inertia_tensor_inv = np.linalg.inv(body_inertia_tensor),
            pose = Pose.from_pq(p = pos),
            linear_momentum = Vec3(lin_momentum),
            angular_momentum = Vec3(ang_momentum),
            collider=BoxCollider.create(extents, Pose.from_pq(p = pos))
        ), 
        visual=vp.box(color = color, size=vp.vector(*extents), make_trail=False, opacity=opacity)
    )

def make_particle(
        name,
        radius=0.5,
        color = vp.color.green,
        opacity=1.0,
        pos = [0, 0, 0],
        lin_momentum = [0, 0, 0],
        mass = 1.0
):
    return Actor(
        name = name,
        phys_obj=Particle(
            name = name,
            inv_mass = 1 / mass,
            pos =Vec3(pos),
            linear_momentum = Vec3(lin_momentum),
            collider=SphereCollider.create(radius, pos)
        ), 
        visual=vp.sphere(radius = radius, make_trail=False, opacity=opacity)
    )