import importlib
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent


from solutions.beads import optimal_bead_config
from lib.beads import build_necklace


TOL = 1e-5
COLLISION_MARGIN = 2.0 - 1e-4
JOINT_LIMIT_RADIUS = np.pi / 3

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
    vertices = build_necklace(link_lengths, angles)
    n_joints = len(link_lengths) - 1
    assert angles.shape == (n_joints, 2), f"Expected shape {(n_joints, 2)}, got {angles.shape}"

    cos_limit = np.cos(JOINT_LIMIT_RADIUS + TOL)
    for i in range(n_joints):
        ax, ay = angles[i, 0], angles[i, 1]
        cos_angle = np.cos(ax) * np.cos(ay)
        assert cos_angle >= cos_limit, (
            f"Joint {i} violates spherical cap: cos(θ)={cos_angle:.4f} < cos(limit)={cos_limit:.4f}"
        )

    n_points = len(vertices)
    for i in range(n_points):
        for j in range(i + 2, n_points):
            d = float(np.linalg.norm(vertices[i] - vertices[j]))
            assert d >= COLLISION_MARGIN, f"Sphere pair ({i},{j}) collides, distance {d:.4f} < {COLLISION_MARGIN}"

@pytest.mark.skipif(REFERENCE_SOLUTION is None, reason="Reference solution is not available")
@pytest.mark.parametrize("link_lengths", OPEN_CASES + (HIDDEN_CASES or []))
def test_improvement_over_reference(link_lengths: np.ndarray) -> None:
    solution_angles = optimal_bead_config(link_lengths)
    solution_vertices = build_necklace(link_lengths, solution_angles)
    solution_radius = np.linalg.norm(solution_vertices - solution_vertices.mean(axis=0), axis=1).max()
    reference_angles = REFERENCE_SOLUTION(link_lengths)
    reference_vertices = build_necklace(link_lengths, reference_angles)
    reference_radius = np.linalg.norm(reference_vertices - reference_vertices.mean(axis=0), axis=1).max()
    assert solution_radius <= reference_radius + TOL, f"""Bounding sphere radius {solution_radius:.4f} > reference {reference_radius:.4f}, 
                                                          min link length {link_lengths.min():.4f}, max link length {link_lengths.max():.4f}, links count {len(link_lengths)}"""
