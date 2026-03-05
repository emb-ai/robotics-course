from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Configuration:
    x: float
    y: float
    z: float
    theta: float
    phi: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def direction(self) -> np.ndarray:
        ct, st = np.cos(self.theta), np.sin(self.theta)
        cp, sp = np.cos(self.phi), np.sin(self.phi)
        return np.array([ct * cp, st * cp, sp])


@dataclass
class XYZConfiguration:
    x: float
    y: float
    z: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


KAPPA_MAX = 1.0
PHI_MIN = -np.pi / 4
PHI_MAX = np.pi / 4
V = 1.0


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
        kappa_worst = np.max(kappa)
        if kappa_worst > kappa_max + tol:
            errors.append(f"Curvature exceeded: max(kappa)={kappa_worst:.4f} > {kappa_max}")

    speed = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2) / ds
    speed_dev = np.max(np.abs(speed - V))
    if speed_dev > 0.1:
        errors.append(f"Speed deviation from v={V}: max|v-1|={speed_dev:.4f}")

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

    d_start = np.linalg.norm(
        np.array([c0.x - start.x, c0.y - start.y, c0.z - start.z])
    )
    if d_start > pos_tol:
        errors.append(f"Start position error: {d_start:.4f}")
    for attr in ("theta", "phi"):
        diff = abs(getattr(c0, attr) - getattr(start, attr))
        if diff > angle_tol:
            errors.append(f"Start {attr} error: {diff:.4f}")

    if goal is not None:
        d_goal = np.linalg.norm(
            np.array([c1.x - goal.x, c1.y - goal.y, c1.z - goal.z])
        )
        if d_goal > pos_tol:
            errors.append(f"Goal position error: {d_goal:.4f}")
        for attr in ("theta", "phi"):
            diff = abs(getattr(c1, attr) - getattr(goal, attr))
            if diff > angle_tol:
                errors.append(f"Goal {attr} error: {diff:.4f}")
    elif goal_xyz is not None:
        d_goal = np.linalg.norm(
            np.array([c1.x - goal_xyz.x, c1.y - goal_xyz.y, c1.z - goal_xyz.z])
        )
        if d_goal > pos_tol:
            errors.append(f"Goal XYZ error: {d_goal:.4f}")

    return len(errors) == 0, errors


def check_all(
    curve_fn: Callable[[np.ndarray], Configuration],
    start: Configuration,
    goal: Configuration | None = None,
    goal_xyz: XYZConfiguration | None = None,
    n_points: int = 500,
) -> tuple[bool, list[str]]:
    all_errors: list[str] = []
    ok1, e1 = check_eom(curve_fn, n_points)
    all_errors.extend(e1)
    ok2, e2 = check_constraints(curve_fn, n_points)
    all_errors.extend(e2)
    ok3, e3 = check_endpoints(curve_fn, start, goal=goal, goal_xyz=goal_xyz)
    all_errors.extend(e3)
    return len(all_errors) == 0, all_errors


def curve_length(
    curve_fn: Callable[[np.ndarray], Configuration],
    n_points: int = 1000,
) -> float:
    s = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs = [], [], []
    for si in s:
        c = curve_fn(np.atleast_1d(si))
        xs.append(c.x)
        ys.append(c.y)
        zs.append(c.z)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    return float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2)))
