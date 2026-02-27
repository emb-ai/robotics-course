from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import scipy.linalg
from IPython.display import HTML

from .cartpole_sim import (
    CartPoleParams, DEFAULT_PARAMS,
    cartpole_dynamics, simulate_cartpole, linearize_cartpole,
    solve_lqr, cartpole_energy, upright_energy, animate_cartpole,
)


def _simulate_linear_2d(
    A: NDArray, x0: NDArray, T: float = 6.0, dt: float = 0.01,
) -> tuple[NDArray, NDArray]:
    ts = np.arange(0, T, dt)
    xs = np.zeros((len(ts), 2))
    xs[0] = x0
    for i in range(len(ts) - 1):
        xs[i + 1] = xs[i] + dt * (A @ xs[i])
    return ts, xs


def show_lyapunov_concept():
    A = np.array([[-1.0, 0.5], [-0.5, -1.0]])
    Q = np.eye(2)
    P = np.array(scipy.linalg.solve_continuous_lyapunov(A.T, -Q))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: contours + vector field + trajectory ---
    ax = axes[0]
    grid = np.linspace(-3, 3, 40)
    X1, X2 = np.meshgrid(grid, grid)
    V = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            xv = np.array([X1[i, j], X2[i, j]])
            V[i, j] = xv @ P @ xv

    ax.contour(X1, X2, V, levels=12, cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("$V(x) = x^\\top P x$ contours + vector field")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    grid_arrow = np.linspace(-3, 3, 16)
    XA, YA = np.meshgrid(grid_arrow, grid_arrow)
    UA = A[0, 0] * XA + A[0, 1] * YA
    VA_field = A[1, 0] * XA + A[1, 1] * YA
    ax.quiver(XA, YA, UA, VA_field, color="gray", alpha=0.5, scale=30)

    for x0 in [np.array([2.5, 2.0]), np.array([-2.0, 2.5]), np.array([1.5, -2.5])]:
        ts, xs = _simulate_linear_2d(A, x0)
        ax.plot(xs[:, 0], xs[:, 1], lw=1.5, zorder=5)
        ax.plot(x0[0], x0[1], "o", markersize=5, zorder=6)

    # --- right: V(t) along trajectories ---
    ax2 = axes[1]
    for x0 in [np.array([2.5, 2.0]), np.array([-2.0, 2.5]), np.array([1.5, -2.5])]:
        ts, xs = _simulate_linear_2d(A, x0)
        Vt = np.array([x @ P @ x for x in xs])
        ax2.plot(ts, Vt, lw=1.5)
    ax2.set_xlabel("t")
    ax2.set_ylabel("$V(x(t))$")
    ax2.set_title("Lyapunov function decreasing along trajectories")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_energy_landscape(p: CartPoleParams = DEFAULT_PARAMS):
    theta_range = np.linspace(-np.pi, np.pi, 200)
    tdot_range = np.linspace(-8, 8, 200)
    TH, TD = np.meshgrid(theta_range, tdot_range)

    E = np.zeros_like(TH)
    for i in range(TH.shape[0]):
        for j in range(TH.shape[1]):
            state = np.array([0.0, TH[i, j], 0.0, TD[i, j]])
            E[i, j] = cartpole_energy(state, p)

    E_target = upright_energy(p)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cf = ax.contourf(TH, TD, E, levels=30, cmap="viridis")
    ax.contour(TH, TD, E, levels=[E_target], colors="red", linewidths=2)
    fig.colorbar(cf, ax=ax, label="Energy")
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    ax.set_title("Cartpole energy landscape")
    ax.plot(0, 0, "r*", markersize=14, label="upright equilibrium")
    ax.legend(fontsize=8)

    # energy controller direction field
    ax2 = axes[1]
    k_e = 20.0
    grid_th = np.linspace(-np.pi, np.pi, 20)
    grid_td = np.linspace(-8, 8, 20)
    TH_g, TD_g = np.meshgrid(grid_th, grid_td)
    U_arrow = np.zeros_like(TH_g)
    for i in range(TH_g.shape[0]):
        for j in range(TH_g.shape[1]):
            state = np.array([0.0, TH_g[i, j], 0.0, TD_g[i, j]])
            E_cur = cartpole_energy(state, p)
            u = k_e * (E_cur - E_target) * TD_g[i, j] * np.cos(TH_g[i, j])
            U_arrow[i, j] = u

    ax2.contourf(TH, TD, E, levels=30, cmap="viridis", alpha=0.4)
    ax2.contour(TH, TD, E, levels=[E_target], colors="red", linewidths=2)
    im = ax2.pcolormesh(TH_g, TD_g, U_arrow, cmap="RdBu", shading="auto", alpha=0.6)
    fig.colorbar(im, ax=ax2, label="Control u")
    ax2.set_xlabel(r"$\theta$ [rad]")
    ax2.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    ax2.set_title("Energy-pumping controller direction")

    fig.tight_layout()
    plt.show()


def show_energy_swingup_demo(p: CartPoleParams = DEFAULT_PARAMS):
    A, B = linearize_cartpole(p)
    Q = np.diag([1.0, 10.0, 1.0, 10.0])
    R = np.array([[1.0]])
    K, _ = solve_lqr(A, B, Q, R)

    E_target = upright_energy(p)
    k_e = 20.0
    k_x = 1.0
    k_xd = 1.0
    theta_thresh = 0.3
    tdot_thresh = 2.0

    switch_times: list[float] = []

    def controller(t: float, state: NDArray) -> float:
        theta = state[1]
        theta_dot = state[3]
        x, x_dot = state[0], state[2]
        near_upright = abs(theta) < theta_thresh and abs(theta_dot) < tdot_thresh
        if near_upright:
            if not switch_times or (switch_times[-1] < 0):
                switch_times.append(t)
            u = float(-K @ state)
        else:
            if switch_times and switch_times[-1] > 0:
                switch_times.append(-t)
            E = cartpole_energy(state, p)
            u = k_e * (E - E_target) * theta_dot * np.cos(theta) - k_x * x - k_xd * x_dot
        return np.clip(u, -p.f_max, p.f_max)

    state0 = np.array([0.0, np.pi - 0.05, 0.0, 0.0])
    ts, states, controls = simulate_cartpole(state0, controller, T=10.0, dt=0.02, p=p)

    energies = np.array([cartpole_energy(s, p) for s in states])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(ts, states[:, 1], label=r"$\theta$")
    ax.plot(ts, states[:, 0], label="x")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    for st in switch_times:
        if st > 0:
            ax.axvline(st, color="green", ls="--", alpha=0.6, label="LQR switch")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("state")
    ax.set_title("State trajectories")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ts, energies, label="E(t)")
    ax.axhline(E_target, color="red", ls="--", label="$E_{target}$")
    for st in switch_times:
        if st > 0:
            ax.axvline(st, color="green", ls="--", alpha=0.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Energy")
    ax.set_title("Energy over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ts, controls)
    for st in switch_times:
        if st > 0:
            ax.axvline(st, color="green", ls="--", alpha=0.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("u [N]")
    ax.set_title("Control input")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(states[:, 1], states[:, 3], lw=0.8)
    ax.plot(states[0, 1], states[0, 3], "go", markersize=8, label="start")
    ax.plot(0, 0, "r*", markersize=12, label="target")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_title("Phase portrait")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

    return animate_cartpole(ts, states, controls, p)


def show_region_of_attraction(p: CartPoleParams = DEFAULT_PARAMS):
    A, B = linearize_cartpole(p)
    Q = np.diag([1.0, 10.0, 1.0, 10.0])
    R = np.array([[1.0]])
    K, _ = solve_lqr(A, B, Q, R)

    def lqr_controller(_t: float, state: NDArray) -> float:
        return float(-K @ state)

    theta_range = np.linspace(-np.pi, np.pi, 60)
    tdot_range = np.linspace(-10, 10, 60)
    stabilized = np.zeros((len(tdot_range), len(theta_range)))

    for i, td in enumerate(tdot_range):
        for j, th in enumerate(theta_range):
            state0 = np.array([0.0, th, 0.0, td])
            try:
                ts, states, _ = simulate_cartpole(state0, lqr_controller, T=5.0, dt=0.02, p=p)
                final = states[-1]
                if abs(final[1]) < 0.05 and abs(final[3]) < 0.5:
                    stabilized[i, j] = 1.0
            except Exception:
                stabilized[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(
        theta_range, tdot_range, stabilized,
        cmap="RdYlGn", shading="auto", alpha=0.8,
    )
    ax.contour(theta_range, tdot_range, stabilized, levels=[0.5], colors="black", linewidths=1.5)

    # overlay a few trajectories
    for th0 in [-1.0, -0.5, 0.5, 1.0]:
        for td0 in [-4.0, 0.0, 4.0]:
            state0 = np.array([0.0, th0, 0.0, td0])
            ts, states, _ = simulate_cartpole(state0, lqr_controller, T=5.0, dt=0.02, p=p)
            ax.plot(states[:, 1], states[:, 3], "k-", lw=0.4, alpha=0.6)

    ax.plot(0, 0, "r*", markersize=14, zorder=10)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    ax.set_title("Region of attraction for LQR around upright equilibrium")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
