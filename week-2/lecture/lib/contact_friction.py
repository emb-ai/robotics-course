from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Contact normal force models
# ---------------------------------------------------------------------------

def penalty_normal_force(
    penetration: float, penetration_rate: float,
    k: float = 1e4, c: float = 1e2,
) -> float:
    if penetration <= 0:
        return 0.0
    return k * penetration + c * max(penetration_rate, 0.0)


def hertz_normal_force(
    penetration: float, penetration_rate: float,
    k: float = 1e4, c: float = 1e2, exponent: float = 1.5,
) -> float:
    if penetration <= 0:
        return 0.0
    elastic = k * penetration ** exponent
    damping = c * penetration ** exponent * max(penetration_rate, 0.0)
    return elastic + damping


# ---------------------------------------------------------------------------
# Friction models
# ---------------------------------------------------------------------------

def coulomb_friction_force(
    v_tangent: Vec3, F_normal_mag: float, mu: float,
) -> Vec3:
    speed = np.linalg.norm(v_tangent)
    if speed < 1e-14:
        return np.zeros(3)
    return -mu * F_normal_mag * v_tangent / speed


def regularized_coulomb_friction(
    v_tangent: Vec3, F_normal_mag: float, mu: float,
    eps: float = 1e-3,
) -> Vec3:
    speed = np.linalg.norm(v_tangent)
    return -mu * F_normal_mag * v_tangent / max(speed, eps)


def viscous_friction(v_tangent: Vec3, b: float) -> Vec3:
    return -b * v_tangent


def tanh_friction(
    v_tangent: Vec3, F_normal_mag: float, mu: float,
    v_scale: float = 0.01,
) -> Vec3:
    speed = np.linalg.norm(v_tangent)
    if speed < 1e-14:
        return np.zeros(3)
    direction = v_tangent / speed
    magnitude = mu * F_normal_mag * np.tanh(speed / v_scale)
    return -magnitude * direction


# ---------------------------------------------------------------------------
# Friction cone / pyramid utilities
# ---------------------------------------------------------------------------

def is_inside_friction_cone(F_tangent: Vec3, F_normal_mag: float, mu: float) -> bool:
    return np.linalg.norm(F_tangent) <= mu * F_normal_mag + 1e-12


def friction_pyramid_directions_2d() -> NDArray:
    return np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def friction_pyramid_directions_3d(n_edges: int = 4) -> NDArray:
    angles = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)
    dirs = np.zeros((n_edges, 3))
    dirs[:, 0] = np.cos(angles)
    dirs[:, 1] = np.sin(angles)
    return dirs


# ---------------------------------------------------------------------------
# Collision impulse (coefficient-of-restitution model)
# ---------------------------------------------------------------------------

def collision_impulse_particle(
    v_rel_normal: float, m_eff: float, restitution: float,
) -> float:
    if v_rel_normal >= 0:
        return 0.0
    return -(1 + restitution) * m_eff * v_rel_normal


def collision_impulse_rigid_body(
    v_rel: Vec3, normal: Vec3, r_a: Vec3, r_b: Vec3,
    m_a: float, m_b: float,
    I_a_inv: NDArray, I_b_inv: NDArray,
    restitution: float,
) -> float:
    v_rel_n = np.dot(v_rel, normal)
    if v_rel_n >= 0:
        return 0.0

    # effective mass at contact point
    ra_cross_n = np.cross(r_a, normal)
    rb_cross_n = np.cross(r_b, normal)
    inv_m_eff = (
        1.0 / m_a + 1.0 / m_b
        + np.dot(normal, np.cross(I_a_inv @ ra_cross_n, r_a))
        + np.dot(normal, np.cross(I_b_inv @ rb_cross_n, r_b))
    )
    return -(1 + restitution) * v_rel_n / inv_m_eff


def apply_impulse(
    p: Vec3, L: Vec3, J: float, normal: Vec3, r_contact: Vec3,
) -> tuple[Vec3, Vec3]:
    p_new = p + J * normal
    L_new = L + np.cross(r_contact, J * normal)
    return p_new, L_new


# ---------------------------------------------------------------------------
# Simple 1D bouncing ball simulation (for demo)
# ---------------------------------------------------------------------------

def simulate_bouncing_ball_penalty(
    y0: float, vy0: float, dt: float, n_steps: int,
    g: float = 9.81, k: float = 1e4, c: float = 100.0, m: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray]:
    ys = np.empty(n_steps)
    vs = np.empty(n_steps)
    ts = np.arange(n_steps) * dt
    y, vy = y0, vy0
    for i in range(n_steps):
        ys[i], vs[i] = y, vy
        penetration = -y
        pen_rate = -vy
        F_contact = penalty_normal_force(penetration, pen_rate, k=k, c=c)
        a = -g + F_contact / m
        vy += a * dt
        y += vy * dt
    return ts, ys, vs


def simulate_bouncing_ball_impulse(
    y0: float, vy0: float, dt: float, n_steps: int,
    g: float = 9.81, restitution: float = 0.8, m: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray]:
    ys = np.empty(n_steps)
    vs = np.empty(n_steps)
    ts = np.arange(n_steps) * dt
    y, vy = y0, vy0
    for i in range(n_steps):
        ys[i], vs[i] = y, vy
        vy -= g * dt
        y += vy * dt
        if y < 0:
            y = 0.0
            if vy < 0:
                vy = -restitution * vy
    return ts, ys, vs
