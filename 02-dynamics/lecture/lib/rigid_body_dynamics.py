from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Vec3 = NDArray[np.float64]  # (3,)
Quat = NDArray[np.float64]  # (4,)  — [w, x, y, z]
Mat3 = NDArray[np.float64]  # (3, 3)


# ---------------------------------------------------------------------------
# Quaternion utilities (Hamilton convention: q = w + xi + yj + zk)
# ---------------------------------------------------------------------------

def quat_multiply(q1: Quat, q2: Quat) -> Quat:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: Quat) -> Quat:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: Quat) -> Quat:
    return q / np.linalg.norm(q)


def quat_to_rotation_matrix(q: Quat) -> Mat3:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_derivative(q: Quat, omega_body: Vec3) -> Quat:
    omega_quat = np.array([0.0, *omega_body])
    return 0.5 * quat_multiply(q, omega_quat)


# ---------------------------------------------------------------------------
# Inertia tensor helpers
# ---------------------------------------------------------------------------

def inertia_tensor_box(m: float, lx: float, ly: float, lz: float) -> Mat3:
    return (m / 12) * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])


def inertia_tensor_sphere(m: float, r: float) -> Mat3:
    return (2 / 5) * m * r**2 * np.eye(3)


def inertia_tensor_cylinder(m: float, r: float, h: float) -> Mat3:
    Ixx = (m / 12) * (3 * r**2 + h**2)
    Izz = (m / 2) * r**2
    return np.diag([Ixx, Ixx, Izz])


def parallel_axis(I_cm: Mat3, m: float, d: Vec3) -> Mat3:
    return I_cm + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


# ---------------------------------------------------------------------------
# Rigid body state  —  (x_cm, q, p, L)  packed into a 13-vector
# ---------------------------------------------------------------------------
#   x_cm : [0:3]     position of center of mass
#   q    : [3:7]     orientation quaternion  [w, x, y, z]
#   p    : [7:10]    linear momentum
#   L    : [10:13]   angular momentum (world frame)

STATE_DIM = 13


def pack_state(x_cm: Vec3, q: Quat, p: Vec3, L: Vec3) -> NDArray:
    return np.concatenate([x_cm, q, p, L])


def unpack_state(state: NDArray) -> tuple[Vec3, Quat, Vec3, Vec3]:
    return state[0:3].copy(), state[3:7].copy(), state[7:10].copy(), state[10:13].copy()


def rigid_body_rhs(
    state: NDArray,
    mass: float,
    I_body: Mat3,
    force_ext: Vec3,
    torque_ext: Vec3,
) -> NDArray:
    x_cm, q, p, L = unpack_state(state)
    q = quat_normalize(q)
    R = quat_to_rotation_matrix(q)

    I_body_inv = np.linalg.inv(I_body)
    # omega in world frame:  omega = R @ I_body_inv @ R^T @ L
    omega_world = R @ I_body_inv @ R.T @ L
    # convert to body frame for quaternion derivative
    omega_body = R.T @ omega_world

    dx = p / mass
    dq = quat_derivative(q, omega_body)
    dp = force_ext
    dL = torque_ext
    return np.concatenate([dx, dq, dp, dL])


def make_rigid_body_ode(
    mass: float,
    I_body: Mat3,
    force_fn: callable = None,
    torque_fn: callable = None,
):
    if force_fn is None:
        force_fn = lambda t, state: np.array([0.0, 0.0, -mass * 9.81])
    if torque_fn is None:
        torque_fn = lambda t, state: np.zeros(3)

    def ode(t: float, state: NDArray) -> NDArray:
        return rigid_body_rhs(
            state, mass, I_body,
            force_fn(t, state), torque_fn(t, state),
        )
    return ode


# ---------------------------------------------------------------------------
# Euler's rotation equations  (body frame, torque-free or with torque)
# State = [omega_1, omega_2, omega_3]
# ---------------------------------------------------------------------------

def euler_rotation_equations(
    I1: float, I2: float, I3: float,
    torque_body_fn: callable = None,
):
    if torque_body_fn is None:
        torque_body_fn = lambda t, omega: np.zeros(3)

    def rhs(t: float, omega: NDArray) -> NDArray:
        w1, w2, w3 = omega
        tau = torque_body_fn(t, omega)
        return np.array([
            (tau[0] + (I2 - I3) * w2 * w3) / I1,
            (tau[1] + (I3 - I1) * w3 * w1) / I2,
            (tau[2] + (I1 - I2) * w1 * w2) / I3,
        ])
    return rhs
