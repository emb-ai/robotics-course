"""Runner script for the kinetic energy demo. Student code lives in solutions/kin_energy.py.

Run from homework directory:  python scripts/kin_energy.py --solver euler
"""
import sys
from pathlib import Path

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from argparse import ArgumentParser

from solutions.kin_energy import KinEnergyCallback
from solutions.ode_solvers import EulerMethod, RK4Method


def main(solver):
    from lib.sim_world import SimWorld
    from lib.entity import make_box_entity, Actor
    import vpython as vp

    rect_extents = [2, 12, 3]
    rect = make_box_entity(
        "rect",
        extents=rect_extents,
        color=vp.color.red,
        lin_momentum=[0.0, 0.0, 0],
        ang_momentum=[10.0, 10.0, 0],
    )
    entities = [rect]

    vp.graph(scroll=True, xmin=0, xmax=5)
    kin_energy_callback = KinEnergyCallback()
    dt = 1e-2
    sim_world = SimWorld(
        entities=entities,
        dt=dt,
        rate=1 / dt,
        dynamics_solver=solver,
        enable_collisions=False,
        callbacks=[kin_energy_callback],
    )
    sim_world.run(1000)
    print("FINISHED")


SOLVERS = {
    "euler": EulerMethod(),
    "rk4": RK4Method(),
}


def parse_args():
    parser = ArgumentParser(description="Kinetic energy")
    parser.add_argument(
        "--solver",
        choices=SOLVERS.keys(),
        help="Which solver to use",
        default="euler",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(SOLVERS[args.solver])
