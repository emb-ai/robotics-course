from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class PurePursuit:
    """Pure pursuit path tracker for Ackermann/car-like robots.

    Reference: Coulter (1992).
    """
    lookahead: float = 1.5    # metres
    wheelbase: float = 2.7    # metres
    min_speed: float = 0.5    # m/s (constant speed along path)
    max_steer: float = np.deg2rad(35.0)

    def _find_lookahead_point(
        self,
        pose: np.ndarray,
        path: list[np.ndarray],
    ) -> np.ndarray | None:
        """Find first waypoint beyond lookahead distance."""
        x, y = pose[0], pose[1]
        for wp in path:
            if np.hypot(wp[0] - x, wp[1] - y) >= self.lookahead:
                return np.array([wp[0], wp[1]])
        if path:
            return np.array([path[-1][0], path[-1][1]])
        return None

    def command(
        self,
        pose: np.ndarray,
        path: list[np.ndarray],
    ) -> tuple[float, float]:
        """Return (v, delta) for the current pose given reference path."""
        target = self._find_lookahead_point(pose, path)
        if target is None:
            return 0.0, 0.0
        dx = target[0] - pose[0]
        dy = target[1] - pose[1]
        # Transform to robot frame
        th = pose[2]
        local_x =  dx * np.cos(th) + dy * np.sin(th)
        local_y = -dx * np.sin(th) + dy * np.cos(th)
        L = np.hypot(local_x, local_y)
        if L < 1e-9:
            return self.min_speed, 0.0
        curvature = 2.0 * local_y / L**2
        delta = np.arctan(curvature * self.wheelbase)
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        return self.min_speed, delta

    def reached_goal(self, pose: np.ndarray, goal: np.ndarray, tol: float = 0.5) -> bool:
        return np.hypot(pose[0] - goal[0], pose[1] - goal[1]) < tol


@dataclass
class StanleyController:
    """Stanley front-axle path tracker for Ackermann robots.

    Reference: Thrun et al. (2006), DARPA Urban Challenge winner.
    """
    k: float = 1.0             # cross-track gain
    k_soft: float = 0.5        # softening constant (avoids division by zero)
    wheelbase: float = 2.7
    max_steer: float = np.deg2rad(35.0)

    def command(
        self,
        pose: np.ndarray,
        path: list[np.ndarray],
        v: float,
    ) -> tuple[float, float]:
        """Return steering angle delta."""
        if len(path) < 2:
            return v, 0.0

        x, y, th = pose
        # Find nearest path segment
        min_dist = np.inf
        nearest_idx = 0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            seg = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            t = np.clip(np.dot(np.array([x - p1[0], y - p1[1]]), seg) /
                        (np.dot(seg, seg) + 1e-9), 0, 1)
            proj = p1 + t * seg
            d = np.hypot(proj[0] - x, proj[1] - y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i
                nearest_proj = proj

        p1, p2 = path[nearest_idx], path[nearest_idx + 1]
        heading_path = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        heading_error = _wrap_angle(heading_path - th)

        # Cross-track error sign
        cross = ((p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0]))
        cross_track_error = np.sign(cross) * min_dist

        delta = heading_error + np.arctan2(self.k * cross_track_error, self.k_soft + abs(v))
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        return v, delta


@dataclass
class PIDController:
    """PID controller for heading and velocity tracking (diff-drive robot).

    Controls (v, omega) given a target pose or waypoint.
    """
    kp_lin: float = 1.0
    ki_lin: float = 0.0
    kd_lin: float = 0.1
    kp_ang: float = 2.0
    ki_ang: float = 0.0
    kd_ang: float = 0.1
    max_v: float = 1.0
    max_omega: float = 2.0

    _e_lin_prev: float = 0.0
    _e_ang_prev: float = 0.0
    _int_lin: float = 0.0
    _int_ang: float = 0.0

    def reset(self) -> None:
        self._e_lin_prev = 0.0
        self._e_ang_prev = 0.0
        self._int_lin = 0.0
        self._int_ang = 0.0

    def command(
        self,
        pose: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> tuple[float, float]:
        """Return (v, omega) to move toward target pose."""
        dx = target[0] - pose[0]
        dy = target[1] - pose[1]
        dist = np.hypot(dx, dy)
        desired_heading = np.arctan2(dy, dx)
        heading_error = _wrap_angle(desired_heading - pose[2])

        # Linear PID
        self._int_lin += dist * dt
        d_lin = (dist - self._e_lin_prev) / (dt + 1e-9)
        v = (self.kp_lin * dist + self.ki_lin * self._int_lin + self.kd_lin * d_lin)
        self._e_lin_prev = dist

        # Angular PID
        self._int_ang += heading_error * dt
        d_ang = (heading_error - self._e_ang_prev) / (dt + 1e-9)
        omega = (self.kp_ang * heading_error + self.ki_ang * self._int_ang +
                 self.kd_ang * d_ang)
        self._e_ang_prev = heading_error

        v     = float(np.clip(v,     -self.max_v,     self.max_v))
        omega = float(np.clip(omega, -self.max_omega,  self.max_omega))
        return v, omega

    def reached(self, pose: np.ndarray, target: np.ndarray, tol: float = 0.15) -> bool:
        return np.hypot(pose[0] - target[0], pose[1] - target[1]) < tol
