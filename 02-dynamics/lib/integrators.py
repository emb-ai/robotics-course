from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, NamedTuple

StateVec = NDArray[np.float64]
RHSFn = Callable[[float, StateVec], StateVec]
JacFn = Callable[[float, StateVec], NDArray[np.float64]]


class IntegrationResult(NamedTuple):
    t: NDArray[np.float64]
    y: NDArray[np.float64]  # shape (n_steps+1, dim)


# ---------------------------------------------------------------------------
# Explicit (forward) Euler  —  order 1
# ---------------------------------------------------------------------------

def explicit_euler_step(f: RHSFn, t: float, y: StateVec, h: float) -> StateVec:
    return y + h * f(t, y)


def explicit_euler(
    f: RHSFn, y0: StateVec, t_span: tuple[float, float], h: float,
) -> IntegrationResult:
    t0, tf = t_span
    ts = np.arange(t0, tf + h * 0.5, h)
    ys = np.empty((len(ts), len(y0)))
    ys[0] = y0
    for i in range(len(ts) - 1):
        ys[i + 1] = explicit_euler_step(f, ts[i], ys[i], h)
    return IntegrationResult(ts, ys)


# ---------------------------------------------------------------------------
# Semi-implicit (symplectic) Euler  —  order 1, symplectic
# Assumes state = [positions, velocities] with equal-length halves.
# ---------------------------------------------------------------------------

def symplectic_euler_step(
    accel_fn: Callable[[float, StateVec, StateVec], StateVec],
    t: float, x: StateVec, v: StateVec, h: float,
) -> tuple[StateVec, StateVec]:
    v_new = v + h * accel_fn(t, x, v)
    x_new = x + h * v_new
    return x_new, v_new


def symplectic_euler(
    accel_fn: Callable[[float, StateVec, StateVec], StateVec],
    x0: StateVec, v0: StateVec,
    t_span: tuple[float, float], h: float,
) -> tuple[NDArray, NDArray, NDArray]:
    t0, tf = t_span
    ts = np.arange(t0, tf + h * 0.5, h)
    xs = np.empty((len(ts), len(x0)))
    vs = np.empty((len(ts), len(v0)))
    xs[0], vs[0] = x0, v0
    for i in range(len(ts) - 1):
        xs[i + 1], vs[i + 1] = symplectic_euler_step(accel_fn, ts[i], xs[i], vs[i], h)
    return ts, xs, vs


# ---------------------------------------------------------------------------
# RK2 (explicit midpoint)  —  order 2
# ---------------------------------------------------------------------------

def rk2_step(f: RHSFn, t: float, y: StateVec, h: float) -> StateVec:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    return y + h * k2


def rk2(
    f: RHSFn, y0: StateVec, t_span: tuple[float, float], h: float,
) -> IntegrationResult:
    t0, tf = t_span
    ts = np.arange(t0, tf + h * 0.5, h)
    ys = np.empty((len(ts), len(y0)))
    ys[0] = y0
    for i in range(len(ts) - 1):
        ys[i + 1] = rk2_step(f, ts[i], ys[i], h)
    return IntegrationResult(ts, ys)


# ---------------------------------------------------------------------------
# RK4 (classic)  —  order 4
# ---------------------------------------------------------------------------

def rk4_step(f: RHSFn, t: float, y: StateVec, h: float) -> StateVec:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(
    f: RHSFn, y0: StateVec, t_span: tuple[float, float], h: float,
) -> IntegrationResult:
    t0, tf = t_span
    ts = np.arange(t0, tf + h * 0.5, h)
    ys = np.empty((len(ts), len(y0)))
    ys[0] = y0
    for i in range(len(ts) - 1):
        ys[i + 1] = rk4_step(f, ts[i], ys[i], h)
    return IntegrationResult(ts, ys)


# ---------------------------------------------------------------------------
# Implicit (backward) Euler  —  order 1, A-stable / L-stable
# Uses Newton iteration; accepts an optional analytic Jacobian.
# ---------------------------------------------------------------------------

def implicit_euler_step(
    f: RHSFn, t: float, y: StateVec, h: float,
    jac: JacFn | None = None,
    tol: float = 1e-10, max_iter: int = 50,
) -> StateVec:
    t_next = t + h
    y_next = y + h * f(t, y)  # initial guess from explicit Euler
    dim = len(y)

    for _ in range(max_iter):
        residual = y_next - y - h * f(t_next, y_next)
        if np.linalg.norm(residual) < tol:
            break
        if jac is not None:
            J = np.eye(dim) - h * jac(t_next, y_next)
        else:
            J = np.eye(dim) - h * _numerical_jacobian(f, t_next, y_next)
        y_next = y_next - np.linalg.solve(J, residual)
    return y_next


def implicit_euler(
    f: RHSFn, y0: StateVec, t_span: tuple[float, float], h: float,
    jac: JacFn | None = None,
) -> IntegrationResult:
    t0, tf = t_span
    ts = np.arange(t0, tf + h * 0.5, h)
    ys = np.empty((len(ts), len(y0)))
    ys[0] = y0
    for i in range(len(ts) - 1):
        ys[i + 1] = implicit_euler_step(f, ts[i], ys[i], h, jac=jac)
    return IntegrationResult(ts, ys)


# ---------------------------------------------------------------------------
# BDF2 (2-step implicit)  —  order 2, A-stable
# ---------------------------------------------------------------------------

def bdf2(
    f: RHSFn, y0: StateVec, t_span: tuple[float, float], h: float,
    jac: JacFn | None = None,
) -> IntegrationResult:
    t0, tf = t_span
    ts = np.arange(t0, tf + h * 0.5, h)
    ys = np.empty((len(ts), len(y0)))
    ys[0] = y0
    if len(ts) < 2:
        return IntegrationResult(ts, ys)

    # bootstrap second value with implicit Euler
    ys[1] = implicit_euler_step(f, ts[0], ys[0], h, jac=jac)

    dim = len(y0)
    for i in range(1, len(ts) - 1):
        # BDF2: y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2h/3)*f(t_{n+1}, y_{n+1})
        t_next = ts[i + 1]
        rhs_const = (4.0 / 3) * ys[i] - (1.0 / 3) * ys[i - 1]
        y_next = rhs_const + (2 * h / 3) * f(t_next, rhs_const)  # initial guess

        for _ in range(50):
            residual = y_next - rhs_const - (2 * h / 3) * f(t_next, y_next)
            if np.linalg.norm(residual) < 1e-10:
                break
            if jac is not None:
                J = np.eye(dim) - (2 * h / 3) * jac(t_next, y_next)
            else:
                J = np.eye(dim) - (2 * h / 3) * _numerical_jacobian(f, t_next, y_next)
            y_next = y_next - np.linalg.solve(J, residual)
        ys[i + 1] = y_next

    return IntegrationResult(ts, ys)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _numerical_jacobian(
    f: RHSFn, t: float, y: StateVec, eps: float = 1e-7,
) -> NDArray[np.float64]:
    f0 = f(t, y)
    dim = len(y)
    J = np.empty((dim, dim))
    for j in range(dim):
        y_pert = y.copy()
        y_pert[j] += eps
        J[:, j] = (f(t, y_pert) - f0) / eps
    return J
