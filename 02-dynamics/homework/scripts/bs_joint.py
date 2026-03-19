"""VPython demo: two rods with ball-and-socket joint. Run from homework dir: python scripts/bs_joint.py"""
import sys
from pathlib import Path

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

import vpython as vp
import numpy as np
from lib.sim_world import SimWorld
from lib.entity import make_box_entity, Actor
from lib.basic_structs import Vec3
from lib.phys.constraints.constraints import BallAndSocketPoint
from lib.phys.forces import GravityForce
from solutions.ode_solvers import RK4Method
from solutions.constraints import BallAndSocketJoint
from solutions.constraints_manager import ConstraintsManagerImpulseBased


def main():
    box_extents = np.array([1, 6, 2])
    rod1 = make_box_entity(
        "rod1",
        extents=box_extents,
        lin_momentum=[0.0, 0.0, 0],
        ang_momentum=[0.0, 0.000, 0],
    )

    rod2 = make_box_entity(
        "rod2",
        pos=(0, -box_extents[1], 0),
        mass=2.0,
        color=vp.color.purple,
        extents=box_extents,
        lin_momentum=[10.0, 1.0, 0],
        ang_momentum=[0.0, 0.000, 5],
    )

    entities = [rod1, rod2]

    vp.sphere(
        pos=vp.vector(*(box_extents / 2).tolist()),
        radius=0.5,
        color=vp.color.red,
    )

    phys_objects = [entity.phys_obj for entity in entities if isinstance(entity, Actor)]

    constraints = {
        "fixed_point1": BallAndSocketPoint(
            body=rod1.phys_obj,
            body_fixed_point_local=Vec3(box_extents / 2),
        ),
        "b&s_joint": BallAndSocketJoint(
            rod1.phys_obj,
            rod2.phys_obj,
            Vec3(0, -box_extents[1] / 2, 0),
        ),
    }
    constraints_manager = ConstraintsManagerImpulseBased(
        bodies=phys_objects,
        constraints=constraints,
        beta_baumgarte=0.9,
    )

    forces = {
        "gravity": GravityForce(g_vector=Vec3(0, -9.81, 0)),
    }

    dt = 1e-2

    sim_world = SimWorld(
        dt=dt,
        rate=1 / dt,
        entities=entities,
        dynamics_solver=RK4Method(),
        constraints_manager=constraints_manager,
        forces=forces,
        enable_collisions=False,
    )
    sim_world.run(1000)


if __name__ == "__main__":
    main()
