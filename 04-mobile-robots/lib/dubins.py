"""Dubins path solver: shortest bounded-curvature paths between two configurations."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple


def _mod2pi(x: float) -> float:
    return x - 2.0 * np.pi * np.floor(x / (2.0 * np.pi))


class DubinsPath(NamedTuple):
    path_type: str
    lengths: tuple[float, float, float]
    total_length: float


# ── CSC word formulas (Shkel & Lumelsky 2001) ────────────────────────

def _LSL(alpha: float, beta: float, d: float):
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    p_sq = 2 + d * d - 2 * np.cos(alpha - beta) + 2 * d * (sa - sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp = np.arctan2(cb - ca, d + sa - sb)
    t = _mod2pi(-alpha + tmp)
    q = _mod2pi(beta - tmp)
    return t, p, q


def _RSR(alpha: float, beta: float, d: float):
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    p_sq = 2 + d * d - 2 * np.cos(alpha - beta) + 2 * d * (sb - sa)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp = np.arctan2(ca - cb, d - sa + sb)
    t = _mod2pi(alpha - tmp)
    q = _mod2pi(-beta + tmp)
    return t, p, q


def _LSR(alpha: float, beta: float, d: float):
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    p_sq = -2 + d * d + 2 * np.cos(alpha - beta) + 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
    t = _mod2pi(-alpha + tmp)
    q = _mod2pi(-_mod2pi(beta) + tmp)
    return t, p, q


def _RSL(alpha: float, beta: float, d: float):
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    p_sq = -2 + d * d + 2 * np.cos(alpha - beta) - 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
    t = _mod2pi(alpha - tmp)
    q = _mod2pi(beta - tmp)
    return t, p, q


def _RLR(alpha: float, beta: float, d: float):
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    tmp = (6.0 - d * d + 2 * np.cos(alpha - beta) + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2 * np.pi - np.arccos(tmp))
    t = _mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb) + p / 2.0)
    q = _mod2pi(alpha - beta - t + p)
    return t, p, q


def _LRL(alpha: float, beta: float, d: float):
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    tmp = (6.0 - d * d + 2 * np.cos(alpha - beta) + 2 * d * (-sa + sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2 * np.pi - np.arccos(tmp))
    t = _mod2pi(-alpha - np.arctan2(ca - cb, d + sa - sb) + p / 2.0)
    q = _mod2pi(_mod2pi(beta) - alpha - t + p)
    return t, p, q


_PATH_FUNCS = [
    (_LSL, "LSL"), (_RSR, "RSR"), (_LSR, "LSR"),
    (_RSL, "RSL"), (_RLR, "RLR"), (_LRL, "LRL"),
]


@dataclass
class DubinsCurves:
    rho: float = 1.0

    def _normalize(self, start: np.ndarray, goal: np.ndarray):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        d = np.hypot(dx, dy) / self.rho
        theta = np.arctan2(dy, dx)
        alpha = _mod2pi(start[2] - theta)
        beta = _mod2pi(goal[2] - theta)
        return alpha, beta, d, theta

    def shortest(self, start: np.ndarray, goal: np.ndarray) -> DubinsPath | None:
        alpha, beta, d, _ = self._normalize(start, goal)
        best = None
        for fn, name in _PATH_FUNCS:
            result = fn(alpha, beta, d)
            if result is not None:
                t, p, q = result
                L = (abs(t) + abs(p) + abs(q)) * self.rho
                if best is None or L < best.total_length:
                    best = DubinsPath(name, (t, p, q), L)
        return best

    def all_paths(self, start: np.ndarray, goal: np.ndarray) -> list[DubinsPath]:
        alpha, beta, d, _ = self._normalize(start, goal)
        paths = []
        for fn, name in _PATH_FUNCS:
            result = fn(alpha, beta, d)
            if result is not None:
                t, p, q = result
                L = (abs(t) + abs(p) + abs(q)) * self.rho
                paths.append(DubinsPath(name, (t, p, q), L))
        return paths

    def sample_path(
        self,
        start: np.ndarray,
        path: DubinsPath,
        n_points: int = 200,
    ) -> np.ndarray:
        """Sample (n_points, 3) waypoints along a Dubins path."""
        seg_types = list(path.path_type)
        seg_lengths = list(path.lengths)
        total_param = sum(abs(s) for s in seg_lengths)
        if total_param < 1e-12:
            return start.reshape(1, 3)

        points = []
        x, y, th = start
        for seg_type, seg_len in zip(seg_types, seg_lengths):
            n_seg = max(2, int(n_points * abs(seg_len) / total_param))
            for frac in np.linspace(0, 1, n_seg, endpoint=False):
                points.append([x, y, th])
                ds = abs(seg_len) * self.rho / n_seg
                if seg_type == "S":
                    x += ds * np.cos(th)
                    y += ds * np.sin(th)
                elif seg_type == "L":
                    dth = ds / self.rho
                    x += self.rho * (np.sin(th + dth) - np.sin(th))
                    y -= self.rho * (np.cos(th + dth) - np.cos(th))
                    th += dth
                elif seg_type == "R":
                    dth = ds / self.rho
                    x += self.rho * (np.sin(th) - np.sin(th - dth))
                    y -= self.rho * (np.cos(th) - np.cos(th - dth))
                    th -= dth
        points.append([x, y, th])
        return np.array(points)
