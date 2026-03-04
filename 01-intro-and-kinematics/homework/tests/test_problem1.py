import sys
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from reference_solution import constraint_checks, test_cases
from solutions.problem1 import gate_pass, catch_snitch, catch_ball_and_gate

SCALE = 1.0 / 40.0
OPEN_GATE_N = 2
OPEN_SNITCH_N = 2
OPEN_BALL_GATE_N = 1


@pytest.mark.parametrize("c", list(test_cases.gate_pass_cases(scale=SCALE, include_random=False))[:OPEN_GATE_N])
def test_gate_pass_constraints(c: test_cases.GatePassCase) -> None:
    curve = gate_pass(c.start, c.goal)
    ok, errors = constraint_checks.check_all(curve, c.start, goal=c.goal)
    assert ok, errors


@pytest.mark.parametrize("c", list(test_cases.catch_snitch_cases(scale=SCALE, include_random=False))[:OPEN_SNITCH_N])
def test_catch_snitch_constraints(c: test_cases.CatchSnitchCase) -> None:
    curve = catch_snitch(c.start, c.goal_xyz)
    ok, errors = constraint_checks.check_all(curve, c.start, goal_xyz=c.goal_xyz)
    assert ok, errors


@pytest.mark.parametrize("c", list(test_cases.catch_ball_and_gate_cases(scale=SCALE, include_random=False))[:OPEN_BALL_GATE_N])
def test_catch_ball_and_gate_constraints(c: test_cases.CatchBallAndGateCase) -> None:
    curve = catch_ball_and_gate(c.start, c.intermediate_goal, c.final_goal)
    ok, errors = constraint_checks.check_all(
        curve, c.start, goal=c.final_goal
    )
    assert ok, errors


@pytest.mark.parametrize("c", list(test_cases.gate_pass_cases(scale=SCALE, include_random=False))[:OPEN_GATE_N])
def test_gate_pass_length_vs_reference(c: test_cases.GatePassCase) -> None:
    ref_len = test_cases.reference_length_gate_pass(c)
    if ref_len is None:
        pytest.skip("reference implementation not available")
    curve = gate_pass(c.start, c.goal)
    student_len = constraint_checks.curve_length(curve)
    assert float(student_len) <= float(ref_len) + 1e-4


@pytest.mark.parametrize("c", list(test_cases.catch_snitch_cases(scale=SCALE, include_random=False))[:OPEN_SNITCH_N])
def test_catch_snitch_length_vs_reference(c: test_cases.CatchSnitchCase) -> None:
    ref_len = test_cases.reference_length_catch_snitch(c)
    if ref_len is None:
        pytest.skip("reference implementation not available")
    curve = catch_snitch(c.start, c.goal_xyz)
    student_len = constraint_checks.curve_length(curve)
    assert float(student_len) <= float(ref_len) + 1e-4


@pytest.mark.parametrize("c", list(test_cases.catch_ball_and_gate_cases(scale=SCALE, include_random=False))[:OPEN_BALL_GATE_N])
def test_catch_ball_and_gate_length_vs_reference(c: test_cases.CatchBallAndGateCase) -> None:
    ref_len = test_cases.reference_length_catch_ball_and_gate(c)
    if ref_len is None:
        pytest.skip("reference implementation not available")
    curve = catch_ball_and_gate(c.start, c.intermediate_goal, c.final_goal)
    student_len = constraint_checks.curve_length(curve)
    assert float(student_len) <= float(ref_len) + 1e-4
