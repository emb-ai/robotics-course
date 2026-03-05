import sys
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

URDF_PATH = _hw / "assets" / "so101" / "robot.urdf"

NUM_POS_TOL = 2e-3
NUM_ORIENT_TOL = 5e-2
ANA_POS_TOL = 1e-2
ANA_ORIENT_TOL = 5e-2


def _fk_pose(serial_chain, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    th = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, -1))
    T = serial_chain.forward_kinematics(th)
    M = T.get_matrix()[0].detach().numpy()
    return M[:3, 3], M[:3, :3]


def _downturned_rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-c, -s, 0], [-s, c, 0], [0, 0, -1]], dtype=np.float64)


TARGETS = [
    (np.array([0.09, 0.02, -0.13]), 0.39),
    (np.array([0.12, -0.03, -0.12]), -0.56),
    (np.array([0.13, 0.00, -0.15]), 0.09),
    (np.array([0.08, 0.00, -0.08]), 0.27),
    (np.array([0.10, -0.01, -0.12]), 0.11),
    (np.array([0.12, -0.04, -0.13]), -0.60),
]


@pytest.mark.skipif(not URDF_PATH.exists(), reason="SO101 URDF not found")
@pytest.mark.parametrize("position_xyz,yaw", TARGETS)
def test_numerical_ik_fk_consistency(position_xyz: np.ndarray, yaw: float) -> None:
    pk = pytest.importorskip("pytorch_kinematics")
    from solutions.so101_ik import numerical_ik_so101_downturned

    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")

    q = numerical_ik_so101_downturned(position_xyz, yaw, URDF_PATH)
    assert q is not None, f"Numerical IK returned None for pos={position_xyz}, yaw={yaw}"
    pos, R = _fk_pose(serial_chain, q)
    assert np.linalg.norm(pos - position_xyz) < NUM_POS_TOL
    assert np.linalg.norm(R - _downturned_rotation(yaw)) < NUM_ORIENT_TOL


@pytest.mark.skipif(not URDF_PATH.exists(), reason="SO101 URDF not found")
@pytest.mark.parametrize("position_xyz,yaw", TARGETS)
def test_analytical_ik_fk_consistency(position_xyz: np.ndarray, yaw: float) -> None:
    pk = pytest.importorskip("pytorch_kinematics")
    from solutions.so101_ik import analytical_ik_so101_downturned

    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")

    q = analytical_ik_so101_downturned(position_xyz, yaw, URDF_PATH)
    assert q is not None, f"Analytical IK returned None for pos={position_xyz}, yaw={yaw}"
    pos, R = _fk_pose(serial_chain, q)
    assert np.linalg.norm(pos - position_xyz) < ANA_POS_TOL
    assert np.linalg.norm(R - _downturned_rotation(yaw)) < ANA_ORIENT_TOL


@pytest.mark.skipif(not URDF_PATH.exists(), reason="SO101 URDF not found")
def test_analytical_vs_numerical_random() -> None:
    pk = pytest.importorskip("pytorch_kinematics")
    from solutions.so101_ik import analytical_ik_so101_downturned, numerical_ik_so101_downturned

    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")

    rng = np.random.RandomState(42)
    compared = 0
    for _ in range(20):
        position_xyz = np.array([
            rng.uniform(0.07, 0.14),
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.18, -0.06),
        ])
        yaw = rng.uniform(-0.6, 0.6)

        q_ana = analytical_ik_so101_downturned(position_xyz, yaw, URDF_PATH)
        q_num = numerical_ik_so101_downturned(position_xyz, yaw, URDF_PATH)
        if q_ana is None or q_num is None:
            continue
        pos_ana, _ = _fk_pose(serial_chain, q_ana)
        pos_num, _ = _fk_pose(serial_chain, q_num)
        assert np.linalg.norm(pos_ana - pos_num) < ANA_POS_TOL, (
            f"Analytical vs numerical FK mismatch at pos={position_xyz}, yaw={yaw:.3f}"
        )
        compared += 1
    assert compared >= 3, "Too few targets were solvable by both methods"


@pytest.mark.skipif(not URDF_PATH.exists(), reason="SO101 URDF not found")
def test_sympy_formulas_keys_and_eval() -> None:
    pytest.importorskip("pytorch_kinematics")
    import sympy
    from solutions.so101_ik import SO101_JOINT_NAMES, so101_downturned_ik_formulas

    x, y, z, yaw_s = sympy.symbols("x y z yaw", real=True)
    formulas = so101_downturned_ik_formulas(x, y, z, yaw_s)
    assert set(formulas.keys()) == set(SO101_JOINT_NAMES)
    order = [formulas[k] for k in SO101_JOINT_NAMES]
    eval_fn = sympy.lambdify((x, y, z, yaw_s), order, "numpy")
    for position_xyz, yaw in TARGETS[:3]:
        xv, yv, zv = position_xyz
        q = np.array(eval_fn(xv, yv, zv, float(yaw)), dtype=np.float64).ravel()
        assert q.shape == (5,) and np.all(np.isfinite(q))
