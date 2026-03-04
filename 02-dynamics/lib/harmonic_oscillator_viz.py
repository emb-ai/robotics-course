from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, FloatSlider

from .integrators import explicit_euler, symplectic_euler, rk4, implicit_euler


def _harmonic_rhs(omega_sq: float):
    def f(t, y):
        return np.array([y[1], -omega_sq * y[0]])
    return f


def _harmonic_jac(omega_sq: float):
    J = np.array([[0.0, 1.0], [-omega_sq, 0.0]])
    def jac(t, y):
        return J
    return jac


def _harmonic_accel(omega_sq: float):
    def a(t, x, v):
        return -omega_sq * x
    return a


def _exact_solution(x0: float, v0: float, omega: float, ts: np.ndarray):
    A = np.sqrt(x0**2 + (v0 / omega) ** 2)
    phi = np.arctan2(-v0 / omega, x0)
    xs = A * np.cos(omega * ts + phi)
    vs = -A * omega * np.sin(omega * ts + phi)
    return xs, vs


def _energy(xs, vs, omega_sq):
    return 0.5 * (vs**2 + omega_sq * xs**2)


def show_harmonic_oscillator_comparison(
    x0: float = 1.0, v0: float = 0.0, omega: float = 2 * np.pi,
    T: float = 10.0, h: float = 0.05,
):
    omega_sq = omega**2
    t_span = (0.0, T)
    y0 = np.array([x0, v0])
    f = _harmonic_rhs(omega_sq)
    jac = _harmonic_jac(omega_sq)
    accel = _harmonic_accel(omega_sq)

    res_euler = explicit_euler(f, y0, t_span, h)
    ts_se, xs_se, vs_se = symplectic_euler(accel, np.array([x0]), np.array([v0]), t_span, h)
    res_rk4 = rk4(f, y0, t_span, h)
    res_impl = implicit_euler(f, y0, t_span, h, jac=jac)

    t_exact = np.linspace(0, T, 2000)
    x_exact, v_exact = _exact_solution(x0, v0, omega, t_exact)

    methods = [
        ("Exact",            t_exact,        x_exact,              v_exact),
        ("Explicit Euler",   res_euler.t,    res_euler.y[:, 0],    res_euler.y[:, 1]),
        ("Symplectic Euler", ts_se,          xs_se[:, 0],          vs_se[:, 0]),
        ("RK4",              res_rk4.t,      res_rk4.y[:, 0],     res_rk4.y[:, 1]),
        ("Implicit Euler",   res_impl.t,     res_impl.y[:, 0],    res_impl.y[:, 1]),
    ]

    colors = ["#333333", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    ax_phase = fig.add_subplot(gs[0])
    ax_energy = fig.add_subplot(gs[1])

    for (name, t, x, v), color in zip(methods, colors):
        lw = 2.0 if name == "Exact" else 1.2
        ls = "-" if name == "Exact" else "--"
        alpha = 1.0 if name == "Exact" else 0.85
        ax_phase.plot(x, v, ls, lw=lw, color=color, alpha=alpha, label=name)
        E = _energy(x, v, omega_sq)
        ax_energy.plot(t, E, ls, lw=lw, color=color, alpha=alpha, label=name)

    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("v")
    ax_phase.set_title(f"Phase portrait  (h = {h})")
    ax_phase.legend(fontsize=8)
    ax_phase.set_aspect("equal", adjustable="datalim")
    ax_phase.grid(True, alpha=0.3)

    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("E(t)")
    ax_energy.set_title("Energy vs time")
    ax_energy.legend(fontsize=8)
    ax_energy.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_harmonic_oscillator_interactive(
    x0: float = 1.0, v0: float = 0.0, omega: float = 2 * np.pi, T: float = 10.0,
):
    @interact(
        h=FloatSlider(min=0.005, max=0.2, step=0.005, value=0.05, description="h"),
    )
    def _update(h):
        show_harmonic_oscillator_comparison(x0=x0, v0=v0, omega=omega, T=T, h=h)
