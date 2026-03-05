from __future__ import annotations

from pathlib import Path

import numpy as np

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def analytical_ik_so101_downturned(
    position_xyz: np.ndarray,
    yaw: float,
    urdf_path: Path | None = None,
    joint_limits: bool = True,
) -> np.ndarray | None:
    """Evaluate the student's sympy formulas numerically. Provided, not graded."""
    import sympy

    x, y, z, yaw_s = sympy.symbols("x y z yaw", real=True)
    formulas = so101_downturned_ik_formulas(x, y, z, yaw_s)
    order = [formulas[k] for k in SO101_JOINT_NAMES]
    func = sympy.lambdify((x, y, z, yaw_s), order, "numpy")

    xv, yv, zv = np.asarray(position_xyz, dtype=np.float64).ravel()[:3]
    q = np.array(func(xv, yv, zv, float(yaw)), dtype=np.float64).ravel()
    if len(q) != 5 or not np.all(np.isfinite(q)):
        return None

    if joint_limits and urdf_path is not None and urdf_path.exists():
        try:
            import pytorch_kinematics as pk

            chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())
            serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
            low, high = serial_chain.get_joint_limits()
            low, high = np.asarray(low), np.asarray(high)
            if np.any(q < low) or np.any(q > high):
                return None
        except Exception:
            return None
    return q


def so101_downturned_ik_formulas(x, y, z, yaw):
    """Return a dict mapping each joint name to a sympy expression in (x, y, z, yaw).

    Derive link lengths and geometry from the FK diagram (e.g. assets/FK-info.png).
    Keys must be those in SO101_JOINT_NAMES.
    """
    raise NotImplementedError


def numerical_ik_so101_downturned(
    position_xyz: np.ndarray,
    yaw: float,
    urdf_path: Path | None = None,
    *,
    num_retries: int = 50,
    max_iterations: int = 200,
    pos_tolerance: float = 1e-3,
    rot_tolerance: float = 5e-2,
) -> np.ndarray | None:
    """Numerical IK for a downturned SO101 pose using pytorch_kinematics.

    Derive the 4×4 target transform for the downturned end-effector (z down, yaw in plane)
    from the FK image. Build pk.SerialChain from the URDF and solve with pk.PseudoInverseIK.
    Return the best converged joint vector or None.
    """
    raise NotImplementedError
