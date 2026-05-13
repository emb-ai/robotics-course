"""
SO101 downturned IK tests.
Open numerical: FK vs target < NUM_POS_TOL (1e-3), yaw < NUM_ORIENT_TOL (5e-2).
Open analytical: same checks with relaxed ANA_* (diagram model vs URDF FK).
Hidden numerical: same FK/yaw checks on hidden poses, without requiring one reference IK branch.
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

from solutions.so101_ik import analytical_ik_so101_downturned as analytical_solution
from solutions.so101_ik import numerical_ik_so101_downturned as numerical_solution
from solutions.so101_ik import so101_downturned_ik_symbolic as analytical_solution_formulas


_hw = Path(__file__).resolve().parent.parent

URDF_PATH = _hw / "assets" / "so101" / "robot.urdf"

NUM_POS_TOL = 1e-3
NUM_ORIENT_TOL = 5e-2
# Closed-form uses the teaching diagram; FK on the calibrated URDF is typically ~1–1.5 cm off.
ANA_POS_TOL = 1.7e-2
ANA_ORIENT_TOL = 8e-2
ANA_REF_ATOL = 1e-6
ANA_REF_RTOL = 1e-7
ANA_REF_EXTRA_CASES = 48
ANA_REF_RNG_SEED = 2026

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


def _open_case_flags(case: dict) -> tuple[bool, bool]:
    """Open JSON uses solvable_analytical / solvable_numerical (legacy: solvable == analytical)."""
    if "solvable_analytical" in case:
        return bool(case["solvable_analytical"]), bool(case["solvable_numerical"])
    s = bool(case["solvable"])
    return s, s


def _eval_formula_vector(func, pose: np.ndarray, label: str) -> np.ndarray:
    try:
        values = func(*pose)
    except Exception as exc:
        pytest.fail(f"{label} formulas raised {type(exc).__name__} for pose={pose.tolist()}: {exc}")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (len(SO101_JOINT_NAMES),):
        pytest.fail(
            f"{label} formulas returned shape {arr.shape} for pose={pose.tolist()}, "
            f"expected {(len(SO101_JOINT_NAMES),)}"
        )
    return arr


def _analytical_formula_test_poses(reference_func) -> list[np.ndarray]:
    poses = [
        np.asarray(case["pose"], dtype=np.float64)
        for case in OPEN_CASES
        if _open_case_flags(case)[0]
    ]
    rng = np.random.default_rng(ANA_REF_RNG_SEED)
    target_pose_count = len(poses) + ANA_REF_EXTRA_CASES
    attempts = 0
    while len(poses) < target_pose_count:
        attempts += 1
        if attempts > 1000:
            pytest.fail("Could not generate enough finite analytical reference test poses")
        pose = np.array(
            [
                rng.uniform(0.10, 0.28),
                rng.uniform(-0.13, 0.13),
                rng.uniform(0.00, 0.14),
                rng.uniform(-0.60, 0.60),
            ],
            dtype=np.float64,
        )
        ref_values = _eval_formula_vector(reference_func, pose, "reference")
        if np.all(np.isfinite(ref_values)):
            poses.append(pose)
    return poses


@pytest.mark.parametrize(
    "pose, solvable_numerical",
    [(case["pose"], _open_case_flags(case)[1]) for case in OPEN_CASES],
)
def test_numerical_ik_fk_consistency(pose: np.ndarray, solvable_numerical: bool) -> None:
    """
    Note that this test only partially tests the correctness of the numerical IK solution.
    It does so by comparing the forward kinematics of the numerical IK solution to the target pose.
    """
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    x, y, z, yaw = pose
    q = numerical_solution(x, y, z, yaw)
    if solvable_numerical:
        assert q is not None, (
            f"Numerical IK returned None for pose marked solvable_numerical=True "
            f"(tol {NUM_POS_TOL} m pos, {NUM_ORIENT_TOL} rad yaw)"
        )
    else:
        assert q is None, "Numerical IK returned a solution for pose marked solvable_numerical=False"
        return
    q_arr = np.array([q[name] for name in SO101_JOINT_NAMES])
    position, R_actual = forward_kinematics_unpacked(serial_chain, q_arr)
    yaw_actual = _yaw_from_rotation_matrix(R_actual)
    assert np.linalg.norm(position - np.array([x, y, z])) < NUM_POS_TOL
    assert _yaw_difference(yaw_actual, yaw) < NUM_ORIENT_TOL


@pytest.mark.parametrize(
    "pose, solvable_analytical",
    [(case["pose"], _open_case_flags(case)[0]) for case in OPEN_CASES],
)
def test_analytical_ik_fk_consistency(pose: np.ndarray, solvable_analytical: bool) -> None:
    """Analytical IK must return None iff the pose is not diagram-solvable within URDF joint limits."""
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    x, y, z, yaw = pose
    q = analytical_solution(x, y, z, yaw)
    if solvable_analytical:
        assert q is not None, "Analytical IK returned None for pose marked solvable_analytical=True"
    else:
        assert q is None, "Analytical IK returned a solution for pose marked solvable_analytical=False"
        return
    q_arr = np.array([q[name] for name in SO101_JOINT_NAMES])
    position, R_actual = forward_kinematics_unpacked(serial_chain, q_arr)
    yaw_actual = _yaw_from_rotation_matrix(R_actual)
    assert np.linalg.norm(position - np.array([x, y, z])) < ANA_POS_TOL
    assert _yaw_difference(yaw_actual, yaw) < ANA_ORIENT_TOL


@pytest.mark.parametrize(
    "pose, solvable_analytical, solvable_numerical",
    [(c["pose"], *_open_case_flags(c)) for c in OPEN_CASES],
)
def test_analytical_vs_numerical_open(
    pose: np.ndarray,
    solvable_analytical: bool,
    solvable_numerical: bool,
) -> None:
    """Consistency of None vs solution flags; when both succeed, numerical hits target and analytical is near it."""
    x, y, z, yaw = pose
    target = np.array([x, y, z], dtype=np.float64)
    q_ana = analytical_solution(x, y, z, yaw)
    q_num = numerical_solution(x, y, z, yaw)
    assert (q_ana is not None) == solvable_analytical
    assert (q_num is not None) == solvable_numerical
    if not (solvable_analytical and solvable_numerical):
        return
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    pa, Ra = forward_kinematics_unpacked(serial_chain, np.array([q_ana[n] for n in SO101_JOINT_NAMES]))
    pn, Rn = forward_kinematics_unpacked(serial_chain, np.array([q_num[n] for n in SO101_JOINT_NAMES]))
    assert np.linalg.norm(pn - target) < NUM_POS_TOL
    assert _yaw_difference(_yaw_from_rotation_matrix(Rn), yaw) < NUM_ORIENT_TOL
    assert np.linalg.norm(pa - target) < ANA_POS_TOL
    assert _yaw_difference(_yaw_from_rotation_matrix(Ra), yaw) < ANA_ORIENT_TOL
    assert np.linalg.norm(pa - pn) < ANA_POS_TOL + NUM_POS_TOL


@pytest.mark.skipif(HIDDEN_CASES is None, reason="Hidden tests are not available")
@pytest.mark.parametrize("pose, solvable, _q_ref", HIDDEN_CASES or [])
def test_numerical_ik_hidden_fk_consistency(
    pose: np.ndarray,
    solvable: bool,
    _q_ref: np.ndarray,
) -> None:
    x, y, z, yaw = pose[0], pose[1], pose[2], pose[3]
    q_num = numerical_solution(x, y, z, yaw)
    if solvable:
        assert q_num is not None, f"Numerical IK returned None for pose solvable within {NUM_POS_TOL} xyz absolute error and {NUM_ORIENT_TOL} orientation absolute error"
    else:
        assert q_num is None, f"Numerical IK returned solution for unsolvable pose"
        return
    q_num = np.array([q_num[name] for name in SO101_JOINT_NAMES])
    chain = pk.build_chain_from_urdf(open(URDF_PATH, "rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    position, R_actual = forward_kinematics_unpacked(serial_chain, q_num)
    yaw_actual = _yaw_from_rotation_matrix(R_actual)
    assert np.linalg.norm(position - np.array([x, y, z])) < NUM_POS_TOL
    assert _yaw_difference(yaw_actual, yaw) < NUM_ORIENT_TOL


@pytest.mark.skipif(REFERENCE_SOLUTION is None, reason="Reference solution is not available")
def test_analytical_vs_reference_formulas() -> None:
    x, y, z, yaw_s = sympy.symbols("x y z yaw", real=True)
    solution_formulas = analytical_solution_formulas(x, y, z, yaw_s)
    reference_formulas = REFERENCE_SOLUTION(x, y, z, yaw_s)
    difference = set(SO101_JOINT_NAMES) - set(solution_formulas.keys())
    assert not difference, f"Analytical solution did not return following joint names: {difference}"

    solution_func = sympy.lambdify(
        (x, y, z, yaw_s),
        [solution_formulas[name] for name in SO101_JOINT_NAMES],
        "numpy",
    )
    reference_func = sympy.lambdify(
        (x, y, z, yaw_s),
        [reference_formulas[name] for name in SO101_JOINT_NAMES],
        "numpy",
    )

    for pose in _analytical_formula_test_poses(reference_func):
        reference_values = _eval_formula_vector(reference_func, pose, "reference")
        solution_values = _eval_formula_vector(solution_func, pose, "student")
        for idx, name in enumerate(SO101_JOINT_NAMES):
            student = solution_values[idx]
            reference = reference_values[idx]
            abs_error = abs(student - reference)
            tolerance = ANA_REF_ATOL + ANA_REF_RTOL * abs(reference)
            assert np.isfinite(reference), (
                f"Reference formula for joint {name} returned non-finite value "
                f"for pose={pose.tolist()}: reference={reference}"
            )
            assert np.isfinite(student), (
                f"Analytical formula for joint {name} returned non-finite value "
                f"for pose={pose.tolist()}: student={student}, reference={reference}"
            )
            assert abs_error <= tolerance, (
                f"Analytical formula mismatch for joint {name} at pose={pose.tolist()}: "
                f"student={student}, reference={reference}, abs_error={abs_error}, "
                f"tolerance={tolerance}"
            )
