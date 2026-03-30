import importlib
import json
import sys
from pathlib import Path

import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from lib.broom_racing import (
    Configuration,
    XYZConfiguration,
    check_all,
    check_constraints,
    check_eom,
    check_endpoints,
    curve_length,
)
from solutions.broom_racing import gate_pass, catch_snitch, catch_ball_and_gate

OPEN_CASES = json.load(open(_hw / "tests" / "broom_racing_open_test_cases.json", "r"))

GATE_PASS_REF = CATCH_SNITCH_REF = CATCH_BALL_AND_GATE_REF = None
REF_AVAILABLE = False
try:
    _ref_mod = importlib.import_module("reference_solution.broom_racing")
    GATE_PASS_REF = getattr(_ref_mod, "gate_pass_ref", None)
    CATCH_SNITCH_REF = getattr(_ref_mod, "catch_snitch_ref", None)
    CATCH_BALL_AND_GATE_REF = getattr(_ref_mod, "catch_ball_and_gate_ref", None)
    REF_AVAILABLE = (
        GATE_PASS_REF is not None
        and CATCH_SNITCH_REF is not None
        and CATCH_BALL_AND_GATE_REF is not None
    )
except Exception:
    pass

HIDDEN_CASES = None
try:
    HIDDEN_CASES = getattr(
        importlib.import_module("hidden_tests.test_broom_racing"),
        "HIDDEN_CASES",
        None,
    )
except Exception:
    pass

RTOL = 1e-3
LENGTH_MARGIN = 1e-3


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
    student_curve = catch_ball_and_gate(start, intermediate, final)
    ref_curve = CATCH_BALL_AND_GATE_REF(start, intermediate, final)
    ok, errors = check_all(student_curve, start, goal=final)
    assert ok, errors
    s_len = curve_length(student_curve)
    r_len = curve_length(ref_curve)
    assert s_len <= r_len * (1 + RTOL) + LENGTH_MARGIN, (
        f"Student length {s_len:.4f} > ref {r_len:.4f} * (1+{RTOL})"
    )
