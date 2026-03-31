import importlib
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent


from solutions.beads import optimal_bead_config
from lib.beads import (
    BEAD_MIN_NONADJACENT_CENTER_DIST,
    bead_configuration_violations,
    bounding_sphere_radius,
)


TOL = 1e-5
FEASIBILITY_SLACK = 5e-6

OPEN_CASES = [
    np.array([2.0, 2.0, 2.0]),
    np.array([2.0, 2.0, 3.0, 2.0, 2.0]),
    np.array([2.0, 2.0, 3.0, 2.0, 2.0, 3.0]),
    np.full(30, 3.0),
    np.random.Generator(np.random.PCG64(0xBAD_5EED)).uniform(2.0, 4.0, size=30),
    np.random.Generator(np.random.PCG64(0x600D_5EED)).uniform(2.0, 4.0, size=30),
]
try:
    REFERENCE_SOLUTION = getattr(importlib.import_module("reference_solution.beads"), "optimal_bead_config")
    HIDDEN_CASES = getattr(importlib.import_module("hidden_tests.beads"), "HIDDEN_CASES")
except Exception:
    REFERENCE_SOLUTION = None
    HIDDEN_CASES = None



@pytest.mark.parametrize("link_lengths", OPEN_CASES)
def test_joint_limits_and_collisions(link_lengths: np.ndarray) -> None:
    angles = optimal_bead_config(link_lengths)
    violations = bead_configuration_violations(
        link_lengths,
        angles,
        tol=TOL,
        min_center_dist=BEAD_MIN_NONADJACENT_CENTER_DIST,
    )
    assert not violations, "; ".join(violations)

@pytest.mark.skipif(REFERENCE_SOLUTION is None, reason="Reference solution is not available")
@pytest.mark.parametrize("link_lengths", OPEN_CASES + (HIDDEN_CASES or []))
def test_improvement_over_reference(link_lengths: np.ndarray) -> None:
    solution_angles = optimal_bead_config(link_lengths)
    violations = bead_configuration_violations(
        link_lengths,
        solution_angles,
        tol=TOL,
        min_center_dist=BEAD_MIN_NONADJACENT_CENTER_DIST - FEASIBILITY_SLACK,
    )
    assert not violations, "Improvement test requires a feasible config: " + "; ".join(violations)
    solution_radius = bounding_sphere_radius(link_lengths, solution_angles)
    reference_angles = REFERENCE_SOLUTION(link_lengths)
    reference_radius = bounding_sphere_radius(link_lengths, reference_angles)
    assert solution_radius <= reference_radius + TOL, f"""Bounding sphere radius {solution_radius:.4f} > reference {reference_radius:.4f}, 
                                                          min link length {link_lengths.min():.4f}, max link length {link_lengths.max():.4f}, links count {len(link_lengths)}"""
