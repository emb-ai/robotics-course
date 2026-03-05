import sys
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from lib.broom_types import (
    Configuration,
    XYZConfiguration,
    check_eom,
    check_constraints,
    check_endpoints,
)
from solutions.broom_racing import gate_pass, catch_snitch, catch_ball_and_gate

# -- canonical test cases (Euclidean distance between configs > 6) --

GATE_PASS_CASES = [
    (
        Configuration(0, 0, 0, 0, 0),
        Configuration(8, 0, 0, 0, 0),
    ),
    (
        Configuration(0, 0, 0, 0, 0),
        Configuration(6, 3, 0, np.pi / 6, 0),
    ),
]

CATCH_SNITCH_CASES = [
    (
        Configuration(0, 0, 0, 0, 0),
        XYZConfiguration(7, 0, 0),
    ),
    (
        Configuration(0, 0, 0, 0, 0),
        XYZConfiguration(6, 0, 2),
    ),
]

CATCH_BALL_AND_GATE_CASES = [
    (
        Configuration(0, 0, 0, 0, 0),
        XYZConfiguration(7, 0, 0),
        Configuration(14, 0, 0, 0, 0),
    ),
]


# ---------- gate_pass ----------

@pytest.mark.parametrize("start,goal", GATE_PASS_CASES)
def test_gate_pass_eom(start: Configuration, goal: Configuration) -> None:
    curve = gate_pass(start, goal)
    ok, errors = check_eom(curve)
    assert ok, errors


@pytest.mark.parametrize("start,goal", GATE_PASS_CASES)
def test_gate_pass_constraints(start: Configuration, goal: Configuration) -> None:
    curve = gate_pass(start, goal)
    ok, errors = check_constraints(curve)
    assert ok, errors


@pytest.mark.parametrize("start,goal", GATE_PASS_CASES)
def test_gate_pass_endpoints(start: Configuration, goal: Configuration) -> None:
    curve = gate_pass(start, goal)
    ok, errors = check_endpoints(curve, start, goal=goal)
    assert ok, errors


# ---------- catch_snitch ----------

@pytest.mark.parametrize("start,goal_xyz", CATCH_SNITCH_CASES)
def test_catch_snitch_eom(start: Configuration, goal_xyz: XYZConfiguration) -> None:
    curve = catch_snitch(start, goal_xyz)
    ok, errors = check_eom(curve)
    assert ok, errors


@pytest.mark.parametrize("start,goal_xyz", CATCH_SNITCH_CASES)
def test_catch_snitch_constraints(start: Configuration, goal_xyz: XYZConfiguration) -> None:
    curve = catch_snitch(start, goal_xyz)
    ok, errors = check_constraints(curve)
    assert ok, errors


@pytest.mark.parametrize("start,goal_xyz", CATCH_SNITCH_CASES)
def test_catch_snitch_endpoints(start: Configuration, goal_xyz: XYZConfiguration) -> None:
    curve = catch_snitch(start, goal_xyz)
    ok, errors = check_endpoints(curve, start, goal_xyz=goal_xyz)
    assert ok, errors


# ---------- catch_ball_and_gate ----------

@pytest.mark.parametrize("start,intermediate,final", CATCH_BALL_AND_GATE_CASES)
def test_catch_ball_and_gate_eom(
    start: Configuration,
    intermediate: XYZConfiguration,
    final: Configuration,
) -> None:
    curve = catch_ball_and_gate(start, intermediate, final)
    ok, errors = check_eom(curve)
    assert ok, errors


@pytest.mark.parametrize("start,intermediate,final", CATCH_BALL_AND_GATE_CASES)
def test_catch_ball_and_gate_constraints(
    start: Configuration,
    intermediate: XYZConfiguration,
    final: Configuration,
) -> None:
    curve = catch_ball_and_gate(start, intermediate, final)
    ok, errors = check_constraints(curve)
    assert ok, errors


@pytest.mark.parametrize("start,intermediate,final", CATCH_BALL_AND_GATE_CASES)
def test_catch_ball_and_gate_endpoints(
    start: Configuration,
    intermediate: XYZConfiguration,
    final: Configuration,
) -> None:
    curve = catch_ball_and_gate(start, intermediate, final)
    ok, errors = check_endpoints(curve, start, goal=final)
    assert ok, errors
