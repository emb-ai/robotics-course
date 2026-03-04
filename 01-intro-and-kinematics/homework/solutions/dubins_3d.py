from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

N_STARTS = 64


@dataclass
class Pose3D:
    position: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.direction = np.asarray(self.direction, dtype=float).reshape(3)
        n = np.linalg.norm(self.direction)
        if n > 1e-12:
            self.direction = self.direction / n


def csc_forward_direction(phi1: float, psi1: float, phi2: float, psi2: float) -> np.ndarray:
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    cp1, sp1 = np.cos(psi1), np.sin(psi1)
    cp2, sp2 = np.cos(psi2), np.sin(psi2)
    vx = sp2 * (cp1 * c1 * c2 - s1 * s2) + sp1 * cp2 * c1
    vy = sp2 * (cp1 * s1 * c2 + c1 * s2) + sp1 * cp2 * s1
    vz = cp1 * cp2 - sp1 * sp2 * c2
    return np.array([vx, vy, vz])


def csc_forward_position(
    phi1: float, psi1: float, d: float, phi2: float, psi2: float, r: float
) -> np.ndarray:
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    cp1, sp1 = np.cos(psi1), np.sin(psi1)
    cp2, sp2 = np.cos(psi2), np.sin(psi2)
    inner = sp1 * (d + r * sp2) + r * cp1 * (c2 * (1 - cp2) - 1) + r
    x = c1 * inner + r * (cp2 - 1) * s1 * s2
    y = s1 * inner - r * (cp2 - 1) * c1 * s2
    z = cp1 * (d + r * sp2) + r * sp1 * (c2 * (cp2 - 1) + 1)
    return np.array([x, y, z])


def rotation_to_canonical(direction: np.ndarray) -> np.ndarray:
    v = np.asarray(direction, dtype=float).reshape(3)
    v = v / (np.linalg.norm(v) + 1e-14)
    z_canon = np.array([0.0, 0.0, 1.0])
    if np.dot(v, z_canon) > 1 - 1e-10:
        return np.eye(3)
    if np.dot(v, z_canon) < -1 + 1e-10:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(v, z_canon)
    axis = axis / (np.linalg.norm(axis) + 1e-14)
    angle = np.arccos(np.clip(np.dot(v, z_canon), -1.0, 1.0))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


class DubinsPath3D:
    def __init__(
        self,
        phi1: float,
        psi1: float,
        d: float,
        phi2: float,
        psi2: float,
        radius: float,
        R_canon_to_world: np.ndarray,
        start_position: np.ndarray,
    ) -> None:
        self._phi1 = float(phi1)
        self._psi1 = float(psi1)
        self._d = float(d)
        self._phi2 = float(phi2)
        self._psi2 = float(psi2)
        self._r = float(radius)
        self._R = np.asarray(R_canon_to_world, dtype=float).reshape(3, 3)
        self._p0 = np.asarray(start_position, dtype=float).reshape(3)

    @property
    def length(self) -> float:
        return self._r * abs(self._psi1) + self._d + self._r * abs(self._psi2)

    @property
    def d(self) -> float:
        return self._d

    def sample(self, num_points: int) -> np.ndarray:
        if num_points < 2:
            num_points = 2
        L1 = self._r * abs(self._psi1)
        L2 = self._d
        L3 = self._r * abs(self._psi2)
        total = L1 + L2 + L3
        if total < 1e-12:
            p = self._p0.copy()
            v = self._R @ np.array([0.0, 0.0, 1.0])
            out = np.zeros((1, 6))
            out[0, :3] = p
            out[0, 3:] = v
            return out
        positions: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        s_vals = np.linspace(0, total, num_points, endpoint=True)
        for s in s_vals:
            if s <= L1:
                t = (s / L1) * self._psi1 if L1 > 1e-12 else self._psi1
                pos_c = csc_forward_position(self._phi1, t, 0.0, self._phi2, 0.0, self._r)
                dir_c = csc_forward_direction(self._phi1, t, self._phi2, 0.0)
            elif s <= L1 + L2:
                u = (s - L1) / L2 if L2 > 1e-12 else 1.0
                pos_end1 = csc_forward_position(
                    self._phi1, self._psi1, 0.0, self._phi2, 0.0, self._r
                )
                dir_end1 = csc_forward_direction(
                    self._phi1, self._psi1, self._phi2, 0.0
                )
                pos_c = pos_end1 + u * self._d * dir_end1
                dir_c = dir_end1
            else:
                u = (s - L1 - L2) / L3 if L3 > 1e-12 else 1.0
                t = u * self._psi2
                pos_c = csc_forward_position(
                    self._phi1, self._psi1, self._d, self._phi2, t, self._r
                )
                dir_c = csc_forward_direction(
                    self._phi1, self._psi1, self._phi2, t
                )
            pos_w = self._p0 + (self._R @ pos_c)
            dir_w = self._R @ dir_c
            positions.append(pos_w)
            directions.append(dir_w)
        out = np.zeros((len(positions), 6))
        out[:, :3] = np.array(positions)
        out[:, 3:] = np.array(directions)
        return out


def solve_dubins_3d(start: Pose3D, goal: Pose3D, radius: float) -> DubinsPath3D:
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    R = rotation_to_canonical(start.direction)
    goal_pos_canon = R.T @ (goal.position - start.position)
    goal_dir_canon = R.T @ goal.direction
    goal_dir_canon = goal_dir_canon / (np.linalg.norm(goal_dir_canon) + 1e-14)
    scale = max(1.0, np.linalg.norm(goal_pos_canon))
    goal_pos_n = goal_pos_canon / scale
    dist_approx = np.linalg.norm(goal_pos_canon)
    r_n = radius / scale

    def residual(x: np.ndarray) -> np.ndarray:
        phi1, psi1, d_n, phi2, psi2 = x[0], x[1], x[2], x[3], x[4]
        d = d_n * scale
        pos = csc_forward_position(phi1, psi1, d, phi2, psi2, radius)
        pos_n = pos / scale
        dir_vec = csc_forward_direction(phi1, psi1, phi2, psi2)
        err_pos = (pos_n - goal_pos_n).ravel()
        err_dir = (dir_vec - goal_dir_canon).ravel()
        return np.concatenate([err_pos, err_dir])

    best: DubinsPath3D | None = None
    best_len = np.inf
    np.random.seed(42)
    for _ in range(N_STARTS):
        phi1 = np.random.uniform(-np.pi, np.pi)
        psi1 = np.random.uniform(-np.pi, np.pi)
        d_n = np.random.uniform(0.0, max(0.1, 2 * dist_approx / scale))
        phi2 = np.random.uniform(-np.pi, np.pi)
        psi2 = np.random.uniform(-np.pi, np.pi)
        x0 = np.array([phi1, psi1, d_n, phi2, psi2])
        lb = np.array([-2 * np.pi, -2 * np.pi, 0.0, -2 * np.pi, -2 * np.pi])
        ub = np.array([2 * np.pi, 2 * np.pi, max(10.0, 3 * dist_approx / scale), 2 * np.pi, 2 * np.pi])
        res = least_squares(
            residual,
            x0,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
        )
        if res.cost < 1e-6:
            phi1, psi1, d_n, phi2, psi2 = res.x
            d = float(d_n * scale)
            if d < -1e-6:
                continue
            path = DubinsPath3D(
                phi1, psi1, d, phi2, psi2, radius, R, start.position
            )
            if path.length < best_len:
                best_len = path.length
                best = path
    if best is None:
        for _ in range(N_STARTS):
            phi1 = np.random.uniform(-np.pi, np.pi)
            psi1 = np.random.uniform(-0.5, 0.5)
            d_n = dist_approx / scale * np.random.uniform(0.5, 1.5)
            phi2 = np.random.uniform(-np.pi, np.pi)
            psi2 = np.random.uniform(-0.5, 0.5)
            x0 = np.array([phi1, psi1, max(0.0, d_n), phi2, psi2])
            lb = np.array([-2 * np.pi, -2 * np.pi, 0.0, -2 * np.pi, -2 * np.pi])
            ub = np.array([2 * np.pi, 2 * np.pi, max(10.0, 3 * dist_approx / scale), 2 * np.pi, 2 * np.pi])
            res = least_squares(residual, x0, bounds=(lb, ub), method="trf", ftol=1e-12, xtol=1e-12)
            if res.cost < 1e-4:
                phi1, psi1, d_n, phi2, psi2 = res.x
                d = float(d_n * scale)
                if d >= -1e-6:
                    path = DubinsPath3D(phi1, psi1, d, phi2, psi2, radius, R, start.position)
                    if path.length < best_len:
                        best_len = path.length
                        best = path
    if best is None:
        raise RuntimeError("solve_dubins_3d: no valid CSC path found")
    return best
