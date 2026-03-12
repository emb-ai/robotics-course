"""
SO101 downturned IK tests. Open tests: position error < NUM_POS_TOL (1e-3), orientation < NUM_ORIENT_TOL (5e-2).
Hidden tests: joint-space distance to reference < NUM_Q_TOL (2e-2).
"""
import importlib
from pathlib import Path
import json
import numpy as np
import sympy
import pytest
import torch
import pytorch_kinematics as pk
from pytorch_kinematics.transforms import rotation_conversions

from solutions.so101_ik import numerical_ik_so101_downturned as numerical_solution
from solutions.so101_ik import analytical_ik_so101_downturned as analytical_solution
from solutions.so101_ik import so101_downturned_ik_symbolic as analytical_solution_formulas


_hw = Path(__file__).resolve().parent.parent

URDF_PATH = _hw / "assets" / "so101" / "robot.urdf"

NUM_POS_TOL = 1e-3
NUM_ORIENT_TOL = 5e-2
NUM_Q_TOL = 2e-2
ANA_POS_TOL = 2e-2
ANA_ORIENT_TOL = 5e-2
ANA_NUM_TOL = 1e-3

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

OPEN_CASES = json.load(open(_hw / "tests" / "so101_ik_open_test_cases.json", "r"))
try:
    REFERENCE_SOLUTION = getattr(importlib.import_module("reference_solution.so101_ik"), "analytical_ik_so101_downturned_ref")
    HIDDEN_CASES = getattr(importlib.import_module("hidden_tests.test_so101_ik"), "HIDDEN_CASES")
except Exception:
    REFERENCE_SOLUTION = None
    HIDDEN_CASES = None


def forward_kinematics_unpacked(serial_chain, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, -1))
    T = serial_chain.forward_kinematics(th)
    M = T.get_matrix()[0].detach().numpy()
    position = M[:3, 3]
    R = M[:3, :3]
    return position, R


def _yaw_from_rotation_matrix(R: np.ndarray) -> float:
    M = torch.tensor(np.asarray(R, dtype=np.float32)).unsqueeze(0)
    euler = rotation_conversions.matrix_to_euler_angles(M, "YZX")
    return float(euler[0, 1].numpy())


def _yaw_difference(yaw_a: float, yaw_b: float) -> float:
    cos_diff = np.cos(yaw_a) * np.cos(yaw_b) + np.sin(yaw_a) * np.sin(yaw_b)
    return float(np.arccos(np.clip(cos_diff, -1.0, 1.0)))

@pytest.mark.parametrize(
    "pose, solvable", 
    [(case["pose"], case["solvable"]) for case in OPEN_CASES]
)
def test_numerical_ik_fk_consistency(pose: np.ndarray, solvable: bool) -> None:
    """
    Note that this test only partially tests the correctness of the numerical IK solution.
    It does so by comparing the forward kinematics of the numerical IK solution to the target pose.
    """
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    x, y, z, yaw = pose
    q = numerical_solution(x, y, z, yaw)
    if solvable:
        assert q is not None, f"Numerical IK returned None for pose solvable within {NUM_POS_TOL} xyz absolute error and {NUM_ORIENT_TOL} orientation absolute error"
    else:
        assert q is None, f"Numerical IK returned solution for unsolvable pose"
        return
    q_arr = np.array([q[name] for name in SO101_JOINT_NAMES])
    position, R_actual = forward_kinematics_unpacked(serial_chain, q_arr)
    yaw_actual = _yaw_from_rotation_matrix(R_actual)
    assert np.linalg.norm(position - np.array([x, y, z])) < NUM_POS_TOL
    assert _yaw_difference(yaw_actual, yaw) < NUM_ORIENT_TOL


@pytest.mark.parametrize(
    "pose, solvable",
    [(case["pose"], case["solvable"]) for case in OPEN_CASES]
)
def test_analytical_ik_fk_consistency(pose: np.ndarray, solvable: bool) -> None:
    """
    Note that this test only partially tests the correctness of the analytical IK solution.
    It does so by comparing the forward kinematics of the analytical IK solution to the target pose.
    """
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    x, y, z, yaw = pose
    q = analytical_solution(x, y, z, yaw)
    if solvable:
        assert q is not None, f"Analytical IK returned None for pose solvable within {ANA_POS_TOL} xyz absolute error and {ANA_ORIENT_TOL} orientation absolute error"
    else:
        assert q is None, f"Analytical IK returned solution for unsolvable pose"
        return
    q_arr = np.array([q[name] for name in SO101_JOINT_NAMES])
    position, R_actual = forward_kinematics_unpacked(serial_chain, q_arr)
    yaw_actual = _yaw_from_rotation_matrix(R_actual)
    assert np.linalg.norm(position - np.array([x, y, z])) < ANA_POS_TOL
    assert _yaw_difference(yaw_actual, yaw) < ANA_ORIENT_TOL


@pytest.mark.parametrize(
    "pose",
    [(case["pose"]) for case in OPEN_CASES]
)
def test_analytical_vs_numerical(pose) -> None:
    """
    Both IK solutions should achieve the same pose (position and orientation).
    Compares FK of each solution; allows different joint configurations (multiple IK solutions).
    """
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    x, y, z, yaw = pose
    q_ana = analytical_solution(x, y, z, yaw)
    q_num = numerical_solution(x, y, z, yaw)
    which = "analytical" if q_ana is None else "numerical"
    assert (q_ana is not None) == (q_num is not None), f"One {which} IK solution returned None for pose={pose}"
    if q_ana is not None and q_num is not None:
        pos_ana, R_ana = forward_kinematics_unpacked(serial_chain, np.array([q_ana[name] for name in SO101_JOINT_NAMES]))
        pos_num, R_num = forward_kinematics_unpacked(serial_chain, np.array([q_num[name] for name in SO101_JOINT_NAMES]))
        yaw_ana = _yaw_from_rotation_matrix(R_ana)
        yaw_num = _yaw_from_rotation_matrix(R_num)
        assert np.linalg.norm(pos_ana - pos_num) < ANA_POS_TOL, f"Analytical vs numerical position mismatch for pose={pose}"
        assert _yaw_difference(yaw_ana, yaw_num) < ANA_ORIENT_TOL, f"Analytical vs numerical yaw mismatch for pose={pose}"


@pytest.mark.skipif(HIDDEN_CASES is None, reason="Hidden tests are not available")
@pytest.mark.parametrize("pose, solvable, q_ref", HIDDEN_CASES or [])
def test_numerical_vs_reference(pose: np.ndarray, solvable: bool, q_ref: np.ndarray) -> None:
    x, y, z, yaw = pose[0], pose[1], pose[2], pose[3]
    q_num = numerical_solution(x, y, z, yaw)
    if solvable:
        assert q_num is not None, f"Numerical IK returned None for pose solvable within {NUM_POS_TOL} xyz absolute error and {NUM_ORIENT_TOL} orientation absolute error"
    else:
        assert q_num is None, f"Numerical IK returned solution for unsolvable pose"
        return
    q_num = np.array([q_num[name] for name in SO101_JOINT_NAMES])
    assert np.linalg.norm(q_num - q_ref) < NUM_Q_TOL, f"Numerical vs reference joint configuration mismatch for pose={pose}"


@pytest.mark.skipif(REFERENCE_SOLUTION is None, reason="Reference solution is not available")
def test_analytical_vs_reference_formulas() -> None:
    x, y, z, yaw_s = sympy.symbols("x y z yaw", real=True)
    solution_formulas = analytical_solution_formulas(x, y, z, yaw_s)
    reference_formulas = REFERENCE_SOLUTION(x, y, z, yaw_s)
    difference = set(SO101_JOINT_NAMES) - set(solution_formulas.keys())
    assert not difference, f"Analytical solution did not return following joint names: {difference}"
    for name in SO101_JOINT_NAMES:
        assert solution_formulas[name].equals(reference_formulas[name]), f"Symbolic mismatch for joint {name}: solution != reference"


