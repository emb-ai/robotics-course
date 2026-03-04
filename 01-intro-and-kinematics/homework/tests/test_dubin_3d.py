import sys
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from solutions.dubins_3d import (
    Pose3D,
    DubinsPath3D,
    csc_forward_direction,
    csc_forward_position,
    solve_dubins_3d,
)

ATOL = 1e-4
RTOL = 1e-3


def _check_path_properties(
    path: DubinsPath3D,
    start: Pose3D,
    goal: Pose3D,
    radius: float,
    tol_pos: float = ATOL,
    tol_dir: float = ATOL,
) -> None:
    pts = path.sample(101)
    assert pts.shape[1] == 6
    start_pos = pts[0, :3]
    start_dir = pts[0, 3:]
    end_pos = pts[-1, :3]
    end_dir = pts[-1, 3:]
    np.testing.assert_allclose(start_pos, start.position, atol=tol_pos, rtol=RTOL)
    np.testing.assert_allclose(end_pos, goal.position, atol=tol_pos, rtol=RTOL)
    start_dir = start_dir / (np.linalg.norm(start_dir) + 1e-14)
    end_dir = end_dir / (np.linalg.norm(end_dir) + 1e-14)
    np.testing.assert_allclose(
        start_dir,
        start.direction / np.linalg.norm(start.direction),
        atol=tol_dir,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        end_dir,
        goal.direction / np.linalg.norm(goal.direction),
        atol=tol_dir,
        rtol=RTOL,
    )
    assert path.d >= -tol_pos
    L = path.length
    if L > 1e-10:
        n = len(pts)
        expected_gap = L / (n - 1)
        for i in range(n - 1):
            gap = np.linalg.norm(pts[i + 1, :3] - pts[i, :3])
            assert gap <= expected_gap * 1.5 + 1e-8


def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    bc = c - b
    area2 = np.linalg.norm(np.cross(ab, ac))
    if area2 < 1e-14:
        return np.inf
    return np.linalg.norm(ab) * np.linalg.norm(ac) * np.linalg.norm(bc) / (2 * area2)


def test_straight_line() -> None:
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    dist = 5.0
    goal = Pose3D(
        position=np.array([0.0, 0.0, dist]),
        direction=np.array([0.0, 0.0, 1.0]),
    )
    radius = 1.0
    path = solve_dubins_3d(start, goal, radius)
    _check_path_properties(path, start, goal, radius)
    np.testing.assert_allclose(path.length, dist, atol=ATOL, rtol=RTOL)


def test_planar_u_turn() -> None:
    r = 1.0
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.array([2 * r, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, -1.0]),
    )
    path = solve_dubins_3d(start, goal, r)
    _check_path_properties(path, start, goal, r)
    expected_len = np.pi * r
    np.testing.assert_allclose(path.length, expected_len, atol=ATOL, rtol=RTOL)


def test_curvature_constraint() -> None:
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.array([2.0, 1.0, 1.5]),
        direction=np.array([0.5, 0.3, 0.8]),
    )
    goal.direction /= np.linalg.norm(goal.direction)
    radius = 1.0
    path = solve_dubins_3d(start, goal, radius)
    pts = path.sample(51)
    for i in range(len(pts) - 2):
        R = _circumradius(pts[i, :3], pts[i + 1, :3], pts[i + 2, :3])
        if R < 1e-6:
            continue
        curvature = 1.0 / R
        assert curvature <= 1.0 / radius + 1e-4


def test_round_trip() -> None:
    r = 1.0
    np.random.seed(123)
    for _ in range(5):
        phi1 = np.random.uniform(-np.pi, np.pi)
        psi1 = np.random.uniform(-np.pi * 0.8, np.pi * 0.8)
        d = np.random.uniform(0.5, 3.0)
        phi2 = np.random.uniform(-np.pi, np.pi)
        psi2 = np.random.uniform(-np.pi * 0.8, np.pi * 0.8)
        goal_pos = csc_forward_position(phi1, psi1, d, phi2, psi2, r)
        goal_dir = csc_forward_direction(phi1, psi1, phi2, psi2)
        start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
        goal = Pose3D(position=goal_pos, direction=goal_dir)
        path = solve_dubins_3d(start, goal, r)
        _check_path_properties(path, start, goal, r)
        true_len = r * abs(psi1) + d + r * abs(psi2)
        assert path.length <= true_len + ATOL


def test_scale_invariance() -> None:
    lam = 2.0
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.array([1.0, 0.5, 1.0]),
        direction=np.array([0.2, 0.1, 0.97]),
    )
    goal.direction /= np.linalg.norm(goal.direction)
    path1 = solve_dubins_3d(start, goal, 1.0)
    start_s = Pose3D(position=lam * start.position, direction=start.direction)
    goal_s = Pose3D(position=lam * goal.position, direction=goal.direction)
    path2 = solve_dubins_3d(start_s, goal_s, lam * 1.0)
    np.testing.assert_allclose(path2.length, lam * path1.length, atol=ATOL, rtol=RTOL)


def test_rotational_invariance() -> None:
    R_rot = np.array([
        [0.6, -0.8, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 0.0, 1.0],
    ])
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.array([1.0, 0.3, 0.8]),
        direction=np.array([0.1, 0.2, 0.97]),
    )
    goal.direction /= np.linalg.norm(goal.direction)
    path1 = solve_dubins_3d(start, goal, 1.0)
    start_r = Pose3D(position=R_rot @ start.position, direction=R_rot @ start.direction)
    goal_r = Pose3D(position=R_rot @ goal.position, direction=R_rot @ goal.direction)
    path2 = solve_dubins_3d(start_r, goal_r, 1.0)
    np.testing.assert_allclose(path2.length, path1.length, atol=ATOL, rtol=RTOL)


def test_reversal_both_solve() -> None:
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.array([1.5, 0.5, 1.0]),
        direction=np.array([0.3, 0.1, 0.95]),
    )
    path_ab = solve_dubins_3d(start, goal, 1.0)
    _check_path_properties(path_ab, start, goal, 1.0)
    start_rev = Pose3D(position=goal.position, direction=-goal.direction)
    goal_rev = Pose3D(position=start.position, direction=-start.direction)
    path_ba = solve_dubins_3d(start_rev, goal_rev, 1.0)
    assert path_ba.length > 0
    pts = path_ba.sample(11)
    np.testing.assert_allclose(pts[0, :3], start_rev.position, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(pts[-1, :3], goal_rev.position, atol=ATOL, rtol=RTOL)


def test_sample_shape() -> None:
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.array([2.0, 0.0, 1.0]),
        direction=np.array([0.0, 0.0, 1.0]),
    )
    path = solve_dubins_3d(start, goal, 1.0)
    pts = path.sample(20)
    assert pts.shape == (20, 6)
    pts2 = path.sample(3)
    assert pts2.shape[0] >= 2 and pts2.shape[1] == 6


@pytest.mark.parametrize("seed", [1, 7, 13, 42, 99])
def test_random_configurations(seed: int) -> None:
    np.random.seed(seed)
    angle = np.random.uniform(0, 2 * np.pi)
    elev = np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
    dx = np.cos(elev) * np.cos(angle)
    dy = np.cos(elev) * np.sin(angle)
    dz = np.sin(elev)
    start = Pose3D(position=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0]))
    goal = Pose3D(
        position=np.random.uniform(-2, 2, size=3),
        direction=np.array([dx, dy, dz]),
    )
    path = solve_dubins_3d(start, goal, 1.0)
    _check_path_properties(path, start, goal, 1.0)
