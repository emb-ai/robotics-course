"""VPython demo: boxes in a room with penalty contacts. Run from homework dir: python scripts/penalty.py"""
import sys
from pathlib import Path

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

import vpython as vp
import numpy as np
from lib.sim_world import SimWorld
from lib.entity import make_box_entity, Actor
from lib.basic_structs import Vec3, Pose
from lib.phys.forces import GravityForce
from lib.phys.collisions.collision_detector import FCLCollisionDetector
from solutions.ode_solvers import RK4Method
from solutions.penalty import PenaltyMethod


def main():
    side = 20.0
    thk = 0.3
    s2 = 2 * side - thk
    s3 = 2 * side + thk

    wallR = make_box_entity(
        "wallR", opacity=0.2, extents=(thk, s2, s3), color=vp.color.red, pos=(side, 0, 0)
    )
    wallR.phys_obj.inv_mass = 0
    wallR.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))
    wallL = make_box_entity(
        "wallL", opacity=0.2, extents=(thk, s2, s3), color=vp.color.red, pos=(-side, 0, 0)
    )
    wallL.phys_obj.inv_mass = 0
    wallL.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))
    wallB = make_box_entity(
        "wallB", opacity=0.2, extents=(s3, thk, s3), color=vp.color.blue, pos=(0, -side, 0)
    )
    wallB.phys_obj.inv_mass = 0
    wallB.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))
    wallT = make_box_entity(
        "wallT", opacity=0.2, extents=(s3, thk, s3), color=vp.color.blue, pos=(0, side, 0)
    )
    wallT.phys_obj.inv_mass = 0
    wallT.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))
    wallBK = make_box_entity(
        "wallBK", opacity=0.2, extents=(s2, s2, thk), color=vp.color.gray(0.7), pos=(0, 0, -side)
    )
    wallBK.phys_obj.inv_mass = 0
    wallBK.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))
    wallFW = make_box_entity(
        "wallFW", opacity=0.2, extents=(s2, s2, thk), color=vp.color.gray(0.7), pos=(0, 0, side)
    )
    wallFW.phys_obj.inv_mass = 0
    wallFW.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))

    box_extents = np.array([4, 4, 4])
    box1 = make_box_entity(
        "box1",
        extents=box_extents,
        lin_momentum=[20.0, -0.0, 0],
        ang_momentum=[0.0, 0.000, 0],
    )
    box1.phys_obj.pose = Pose.from_pq(p=[2, 2, 0], q=[0.9961947, 0.0871557, 0, 0])
    box2 = make_box_entity(
        "box2",
        extents=box_extents,
        lin_momentum=[-20.0, -0.0, 0],
        ang_momentum=[0.0, 0.000, 0],
    )
    box2.phys_obj.pose = Pose.from_pq(p=[-2, -2, 0], q=[0.3466353, 0, 0, 0.938])

    entities = [box1, box2, wallR, wallL, wallB, wallT, wallBK, wallFW]
    phys_objects = [entity.phys_obj for entity in entities if isinstance(entity, Actor)]

    forces = {
        "gravity": GravityForce(g_vector=Vec3(0, -9.81, 0)),
    }
    contact_handler = PenaltyMethod(k_s=1e2, k_d=1.0)

    dt = 1e-2

    sim_world = SimWorld(
        dt=dt,
        rate=1 / dt,
        entities=entities,
        dynamics_solver=RK4Method(),
        forces=forces,
        enable_collisions=True,
        contact_forces_handler=contact_handler,
        collision_detector=FCLCollisionDetector(
            bodies=phys_objects,
            allowed_collisions=[
                ("wallR", "wallB"),
                ("wallB", "wallR"),
                ("wallR", "wallT"),
                ("wallT", "wallR"),
                ("wallL", "wallB"),
                ("wallB", "wallL"),
                ("wallL", "wallT"),
                ("wallT", "wallL"),
                ("wallBK", "wallR"),
                ("wallR", "wallBK"),
                ("wallBK", "wallL"),
                ("wallL", "wallBK"),
                ("wallBK", "wallT"),
                ("wallT", "wallBK"),
                ("wallBK", "wallB"),
                ("wallB", "wallBK"),
                ("wallFW", "wallR"),
                ("wallR", "wallFW"),
                ("wallFW", "wallL"),
                ("wallL", "wallFW"),
                ("wallFW", "wallT"),
                ("wallT", "wallFW"),
                ("wallFW", "wallB"),
                ("wallB", "wallFW"),
                ("rod1", "rod2"),
                ("rod2", "rod1"),
            ],
        ),
    )
    sim_world.run(1000)
    print("FINISHED")


if __name__ == "__main__":
    main()
