import importlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from lib.broom_racing import (
    Configuration,
    XYZConfiguration,
    KAPPA_MAX,
    PHI_MIN,
    PHI_MAX,
    V,
    curve_length,
    heading_angular_error_rad,
)
from solutions.broom_racing import gate_pass, catch_snitch, catch_ball_and_gate

OPEN_CASES = json.load(open(_hw / "tests" / "broom_racing_open_test_cases.json", "r"))

try:
    _ref_mod = importlib.import_module("reference_solution.broom_racing")
    GATE_PASS_REF = getattr(_ref_mod, "gate_pass_ref", None)
    CATCH_SNITCH_REF = getattr(_ref_mod, "catch_snitch_ref", None)
    CATCH_BALL_AND_GATE_REF = getattr(_ref_mod, "catch_ball_and_gate_ref", None)
    REF_AVAILABLE = GATE_PASS_REF is not None and CATCH_SNITCH_REF is not None and CATCH_BALL_AND_GATE_REF is not None
    HIDDEN_CASES = getattr(importlib.import_module("hidden_tests.test_broom_racing"), "HIDDEN_CASES", None)
except Exception:
    GATE_PASS_REF = CATCH_SNITCH_REF = CATCH_BALL_AND_GATE_REF = None
    REF_AVAILABLE = False
    HIDDEN_CASES = None

RTOL = 1e-3
LENGTH_MARGIN = 1e-3


def _sample_curve(
    curve_fn: Callable[[np.ndarray], Configuration],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs, thetas, phis = [], [], [], [], []
    for si in s:
        c = curve_fn(np.atleast_1d(si))
        xs.append(c.x)
        ys.append(c.y)
        zs.append(c.z)
        thetas.append(c.theta)
        phis.append(c.phi)
    return (
        np.array(xs),
        np.array(ys),
        np.array(zs),
        np.array(thetas),
        np.array(phis),
    )


def check_eom(
    curve_fn: Callable[[np.ndarray], Configuration],
    n_points: int = 500,
    tol: float = 0.05,
) -> tuple[bool, list[str]]:
    xs, ys, zs, thetas, phis = _sample_curve(curve_fn, n_points)
    L = curve_length(curve_fn, n_points)
    ds = L / (n_points - 1)
    errors: list[str] = []
    if ds < 1e-12:
        return True, errors
    dx = np.diff(xs) / ds
    dy = np.diff(ys) / ds
    dz = np.diff(zs) / ds
    ct = np.cos(thetas[:-1])
    st = np.sin(thetas[:-1])
    cp = np.cos(phis[:-1])
    sp = np.sin(phis[:-1])
    res_x = np.abs(dx - V * ct * cp)
    res_y = np.abs(dy - V * st * cp)
    res_z = np.abs(dz - V * sp)
    if np.max(res_x) > tol:
        errors.append(f"EOM x residual max={np.max(res_x):.4f} > {tol}")
    if np.max(res_y) > tol:
        errors.append(f"EOM y residual max={np.max(res_y):.4f} > {tol}")
    if np.max(res_z) > tol:
        errors.append(f"EOM z residual max={np.max(res_z):.4f} > {tol}")
    return len(errors) == 0, errors


def check_constraints(
    curve_fn: Callable[[np.ndarray], Configuration],
    n_points: int = 500,
    kappa_max: float = KAPPA_MAX,
    phi_min: float = PHI_MIN,
    phi_max: float = PHI_MAX,
    tol: float = 1e-3,
) -> tuple[bool, list[str]]:
    xs, ys, zs, thetas, phis = _sample_curve(curve_fn, n_points)
    L = curve_length(curve_fn, n_points)
    ds = L / (n_points - 1)
    errors: list[str] = []
    phi_lo = np.min(phis)
    phi_hi = np.max(phis)
    if phi_lo < phi_min - tol:
        errors.append(f"Pitch below limit: min(phi)={phi_lo:.4f} < {phi_min:.4f}")
    if phi_hi > phi_max + tol:
        errors.append(f"Pitch above limit: max(phi)={phi_hi:.4f} > {phi_max:.4f}")
    if ds > 1e-12:
        dtheta = np.diff(thetas) / ds
        dphi = np.diff(phis) / ds
        cp = np.cos(phis[:-1])
        u1 = dtheta * cp
        u2 = dphi
        kappa = np.sqrt(u1**2 + u2**2)
        if np.max(kappa) > kappa_max + tol:
            errors.append(f"Curvature exceeded: max(kappa)={np.max(kappa):.4f} > {kappa_max}")
        speed = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2) / ds
        if np.max(np.abs(speed - V)) > 0.1:
            errors.append(f"Speed deviation from v={V}: max|v-1|={np.max(np.abs(speed - V)):.4f}")
    return len(errors) == 0, errors


def check_endpoints(
    curve_fn: Callable[[np.ndarray], Configuration],
    start: Configuration,
    goal: Configuration | None = None,
    goal_xyz: XYZConfiguration | None = None,
    pos_tol: float = 0.02,
    angle_tol: float = 0.05,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    c0 = curve_fn(np.atleast_1d(0.0))
    c1 = curve_fn(np.atleast_1d(1.0))
    d_start = np.linalg.norm(np.array([c0.x - start.x, c0.y - start.y, c0.z - start.z]))
    if d_start > pos_tol:
        errors.append(f"Start position error: {d_start:.4f}")
    d_ang0 = heading_angular_error_rad(c0.theta, c0.phi, start.theta, start.phi)
    if d_ang0 > angle_tol:
        errors.append(f"Start heading error: angle={d_ang0:.4f} rad > {angle_tol}")
    if goal is not None:
        d_goal = np.linalg.norm(np.array([c1.x - goal.x, c1.y - goal.y, c1.z - goal.z]))
        if d_goal > pos_tol:
            errors.append(f"Goal position error: {d_goal:.4f}")
        d_ang1 = heading_angular_error_rad(c1.theta, c1.phi, goal.theta, goal.phi)
        if d_ang1 > angle_tol:
            errors.append(f"Goal heading error: angle={d_ang1:.4f} rad > {angle_tol}")
    elif goal_xyz is not None:
        d_goal = np.linalg.norm(np.array([c1.x - goal_xyz.x, c1.y - goal_xyz.y, c1.z - goal_xyz.z]))
        if d_goal > pos_tol:
            errors.append(f"Goal XYZ error: {d_goal:.4f}")
    return len(errors) == 0, errors


def _start_from_case(case: dict) -> Configuration:
    s = case["start"]
    return Configuration(float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]))


def _goal_from_case(case: dict) -> Configuration:
    g = case["goal"]
    return Configuration(float(g[0]), float(g[1]), float(g[2]), float(g[3]), float(g[4]))


def _goal_xyz_from_case(case: dict) -> XYZConfiguration:
    g = case["goal_xyz"]
    return XYZConfiguration(float(g[0]), float(g[1]), float(g[2]))


GATE_PASS_CASES = [c for c in OPEN_CASES if c["task"] == "gate_pass"]
CATCH_SNITCH_CASES = [c for c in OPEN_CASES if c["task"] == "catch_snitch"]
CATCH_BALL_AND_GATE_CASES = [c for c in OPEN_CASES if c["task"] == "catch_ball_and_gate"]


@pytest.mark.parametrize("case", GATE_PASS_CASES)
def test_gate_pass_all(case: dict) -> None:
    start = _start_from_case(case)
    goal = _goal_from_case(case)
    curve = gate_pass(start, goal)
    ok_eom, errors_eom = check_eom(curve)
    ok_constraints, errors_constraints = check_constraints(curve)
    ok_endpoints, errors_endpoints = check_endpoints(curve, start, goal=goal)
    assert ok_eom, f"EOM check failed for gate_pass({start}, {goal}): {errors_eom}"
    assert ok_constraints, f"Constraint check failed for gate_pass({start}, {goal}): {errors_constraints}"
    assert ok_endpoints, f"Endpoints check failed for gate_pass({start}, {goal}): {errors_endpoints}"


@pytest.mark.parametrize("case", CATCH_SNITCH_CASES)
def test_catch_snitch_all(case: dict) -> None:
    start = _start_from_case(case)
    goal_xyz = _goal_xyz_from_case(case)
    curve = catch_snitch(start, goal_xyz)
    ok_eom, errors_eom = check_eom(curve)
    assert ok_eom, f"EOM check failed for catch_snitch({start}, {goal_xyz}): {errors_eom}"
    ok_constraints, errors_constraints = check_constraints(curve)
    assert ok_constraints, f"Constraint check failed for catch_snitch({start}, {goal_xyz}): {errors_constraints}"
    ok_endpoints, errors_endpoints = check_endpoints(curve, start, goal_xyz=goal_xyz)
    assert ok_endpoints, f"Endpoints check failed for catch_snitch({start}, {goal_xyz}): {errors_endpoints}"


@pytest.mark.parametrize("case", CATCH_BALL_AND_GATE_CASES)
def test_catch_ball_and_gate_all(case: dict) -> None:
    start = _start_from_case(case)
    intermediate = _goal_xyz_from_case(case)
    final = _goal_from_case(case)
    curve = catch_ball_and_gate(start, intermediate, final)
    ok_eom, errors_eom = check_eom(curve)
    assert ok_eom, f"EOM check failed for catch_ball_and_gate({start}, {intermediate}, {final}): {errors_eom}"
    ok_constraints, errors_constraints = check_constraints(curve)
    assert ok_constraints, f"Constraint check failed for catch_ball_and_gate({start}, {intermediate}, {final}): {errors_constraints}"
    ok_endpoints, errors_endpoints = check_endpoints(curve, start, goal=final)
    assert ok_endpoints, f"Endpoints check failed for catch_ball_and_gate({start}, {intermediate}, {final}): {errors_endpoints}"


def _all_cases_for_ref() -> list:
    out = list(OPEN_CASES)
    if HIDDEN_CASES:
        out = out + HIDDEN_CASES
    return out


@pytest.mark.skipif(not REF_AVAILABLE, reason="Reference solution not available")
@pytest.mark.parametrize("case", [c for c in _all_cases_for_ref() if c["task"] == "gate_pass"])
def test_gate_pass_length_vs_reference(case: dict) -> None:
    start = _start_from_case(case)
    goal = _goal_from_case(case)
    from lib.broom_racing import check_all
    student_curve = gate_pass(start, goal)
    ref_curve = GATE_PASS_REF(start, goal)
    ok, errors = check_all(student_curve, start, goal=goal)
    assert ok, errors
    s_len = curve_length(student_curve)
    r_len = curve_length(ref_curve)
    assert s_len <= r_len * (1 + RTOL) + LENGTH_MARGIN, (
        f"Student length {s_len:.4f} > ref {r_len:.4f} * (1+{RTOL})"
    )


@pytest.mark.skipif(not REF_AVAILABLE, reason="Reference solution not available")
@pytest.mark.parametrize("case", [c for c in _all_cases_for_ref() if c["task"] == "catch_snitch"])
def test_catch_snitch_length_vs_reference(case: dict) -> None:
    start = _start_from_case(case)
    goal_xyz = _goal_xyz_from_case(case)
    from lib.broom_racing import check_all
    student_curve = catch_snitch(start, goal_xyz)
    ref_curve = CATCH_SNITCH_REF(start, goal_xyz)
    ok, errors = check_all(student_curve, start, goal_xyz=goal_xyz)
    assert ok, errors
    s_len = curve_length(student_curve)
    r_len = curve_length(ref_curve)
    assert s_len <= r_len * (1 + RTOL) + LENGTH_MARGIN, (
        f"Student length {s_len:.4f} > ref {r_len:.4f} * (1+{RTOL})"
    )


@pytest.mark.skipif(not REF_AVAILABLE, reason="Reference solution not available")
@pytest.mark.parametrize("case", [c for c in _all_cases_for_ref() if c["task"] == "catch_ball_and_gate"])
def test_catch_ball_and_gate_length_vs_reference(case: dict) -> None:
    start = _start_from_case(case)
    intermediate = _goal_xyz_from_case(case)
    final = _goal_from_case(case)
    from lib.broom_racing import check_all
    student_curve = catch_ball_and_gate(start, intermediate, final)
    ref_curve = CATCH_BALL_AND_GATE_REF(start, intermediate, final)
    ok, errors = check_all(student_curve, start, goal=final)
    assert ok, errors
    s_len = curve_length(student_curve)
    r_len = curve_length(ref_curve)
    assert s_len <= r_len * (1 + RTOL) + LENGTH_MARGIN, (
        f"Student length {s_len:.4f} > ref {r_len:.4f} * (1+{RTOL})"
    )
