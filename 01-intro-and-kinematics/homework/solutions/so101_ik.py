from pathlib import Path

import numpy as np
import sympy
import pytorch_kinematics as pk

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "so101" / "robot.urdf"


def so101_downturned_ik_symbolic(
    x: sympy.Symbol,
    y: sympy.Symbol,
    z: sympy.Symbol,
    yaw: sympy.Symbol,
) -> dict[str, sympy.Expr]:
    """
    Return a dict mapping each joint name to a sympy expression in (x, y, z, yaw).

    Parameters
    ----------
    x, y, z, yaw : sympy.Symbol
        Symbols for end-effector position and yaw.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a sympy expression (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    raise NotImplementedError


def analytical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float]:
    """
    Evaluate the analytical IK formulas numerically and check joint limits.

    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw angle (radians) in the downturned end-effector plane.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    x_sym, y_sym, z_sym, yaw_sym = sympy.symbols("x y z yaw", real=True)
    formulas = so101_downturned_ik_symbolic(x_sym, y_sym, z_sym, yaw_sym)
    func = sympy.lambdify(
        (x_sym, y_sym, z_sym, yaw_sym),
        [formulas[k] for k in SO101_JOINT_NAMES],
        "numpy",
    )
    q = dict(zip(SO101_JOINT_NAMES, func(x, y, z, yaw)))

    chain = pk.build_chain_from_urdf(open(URDF_PATH, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    low, high = serial_chain.get_joint_limits()
    low, high = np.asarray(low), np.asarray(high)
    for joint in chain.get_joints():
        joint_name = joint.name
        joint_angle = q[joint_name]
        if joint_angle < low[joint_name] or joint_angle > high[joint_name]:
            return None
    return q


def numerical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float] | None:
    """
    Numerical IK for a downturned SO101 pose.

    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw (radians) in the downturned end-effector plane.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    raise NotImplementedError
