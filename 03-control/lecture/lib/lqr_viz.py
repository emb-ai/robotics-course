from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatLogSlider
import scipy.linalg

from .cartpole_sim import (
    DEFAULT_PARAMS, CartPoleParams,
    simulate_cartpole, linearize_cartpole,
    solve_lqr, animate_cartpole,
)


def show_lqr_cartpole(
    Q_diag: tuple[float, ...] = (1, 10, 1, 10),
    R_val: float = 1.0,
    theta0: float = 0.2,
    T: float = 5.0,
    p: CartPoleParams = DEFAULT_PARAMS,
):
    A, B = linearize_cartpole(p)
    Q = np.diag(Q_diag)
    R = np.array([[R_val]])
    K, P = solve_lqr(A, B, Q, R)

    def controller(t, state):
        return float(-K @ state)

    state0 = np.array([0.0, theta0, 0.0, 0.0])
    ts, states, controls = simulate_cartpole(state0, controller, T=T, dt=0.02, p=p)

    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ts, states[:, 0], label="x [m]")
    ax.plot(ts, states[:, 1], label="θ [rad]")
    ax.plot(ts, states[:, 2], label="ẋ [m/s]", alpha=0.6)
    ax.plot(ts, states[:, 3], label="θ̇ [rad/s]", alpha=0.6)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("State")
    ax.set_title("LQR Closed-Loop States")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts, controls, "g", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("u [N]")
    ax.set_title("Control Input")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(eigvals.real, eigvals.imag, s=80, c="red", zorder=5, marker="x", linewidths=2)
    ax.axvline(0, color="gray", lw=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    max_r = max(np.abs(eigvals).max() * 1.3, 1.0)
    ax.set_xlim(-max_r, max_r * 0.3)
    ax.set_ylim(-max_r, max_r)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Closed-Loop Eigenvalues")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.suptitle("LQR on Linearized CartPole", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

    K_str = ", ".join(f"{k:.3f}" for k in K.ravel())
    print(f"LQR gain K = [{K_str}]")
    print(f"Closed-loop eigenvalues: {[f'{v:.3f}' for v in eigvals]}")

    return animate_cartpole(ts, states, controls, p)


def show_lqr_qr_effects_interactive(p: CartPoleParams = DEFAULT_PARAMS):
    A, B = linearize_cartpole(p)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def _update(q1=1.0, q2=10.0, q3=1.0, q4=10.0, R_val=1.0):
        Q = np.diag([q1, q2, q3, q4])
        R = np.array([[R_val]])
        K, _ = solve_lqr(A, B, Q, R)

        def ctrl(t, state):
            return float(-K @ state)

        state0 = np.array([0.0, 0.2, 0.0, 0.0])
        ts, states, controls = simulate_cartpole(state0, ctrl, T=5.0, dt=0.02, p=p)

        A_cl = A - B @ K
        eigvals = np.linalg.eigvals(A_cl)

        for ax in axes:
            ax.clear()

        axes[0].plot(ts, states[:, 0], label="x")
        axes[0].plot(ts, states[:, 1], label="θ")
        axes[0].axhline(0, color="gray", ls="--", lw=0.5)
        axes[0].set_xlabel("t [s]")
        axes[0].set_ylabel("State")
        axes[0].set_title("States")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, controls, "g", lw=1.5)
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel("u [N]")
        axes[1].set_title("Control")
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(eigvals.real, eigvals.imag, s=80, c="red", marker="x", lw=2)
        axes[2].axvline(0, color="gray", lw=0.5)
        axes[2].axhline(0, color="gray", lw=0.5)
        axes[2].set_xlabel("Re(λ)")
        axes[2].set_ylabel("Im(λ)")
        axes[2].set_title("CL Eigenvalues")
        axes[2].grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    slider = lambda val, desc: FloatLogSlider(value=val, base=10, min=-1, max=2, step=0.05, description=desc)
    interact(
        _update,
        q1=slider(1, "Q_x"), q2=slider(10, "Q_θ"),
        q3=slider(1, "Q_ẋ"), q4=slider(10, "Q_θ̇"),
        R_val=FloatLogSlider(value=1, base=10, min=-2, max=2, step=0.05, description="R"),
    )
    plt.show()


def show_riccati_backward(
    N: int = 200,
    dt: float = 0.02,
    p: CartPoleParams = DEFAULT_PARAMS,
):
    A, B = linearize_cartpole(p)
    Q = np.diag([1.0, 10.0, 1.0, 10.0])
    R = np.array([[1.0]])
    R_inv = np.linalg.inv(R)
    S = B @ R_inv @ B.T

    _, P_inf = solve_lqr(A, B, Q, R)

    T_total = N * dt
    ts = np.linspace(T_total, 0, N + 1)
    Ps = np.zeros((N + 1, 4, 4))
    Ks = np.zeros((N + 1, 1, 4))
    Ps[0] = Q

    for i in range(N):
        P = Ps[i]
        dP = -(A.T @ P + P @ A - P @ S @ P + Q)
        Ps[i + 1] = P - dP * dt
        Ks[i] = R_inv @ B.T @ P

    Ks[-1] = R_inv @ B.T @ Ps[-1]

    ts_plot = ts[::-1]
    Ps_plot = Ps[::-1]
    Ks_plot = Ks[::-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    labels_p = ["P(1,1)", "P(2,2)", "P(3,3)", "P(4,4)"]
    for j in range(4):
        ax1.plot(ts_plot, Ps_plot[:, j, j], label=labels_p[j])
        ax1.axhline(P_inf[j, j], color=f"C{j}", ls=":", alpha=0.4)
    ax1.set_ylabel("P diagonal entries")
    ax1.set_title("Riccati Matrix P(t) — Finite Horizon Backward Integration")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for j in range(4):
        ax2.plot(ts_plot, Ks_plot[:, 0, j], label=f"K(1,{j+1})")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("Gain entries")
    ax2.set_title("Time-Varying Gain K(t)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_lqr_vs_pid(p: CartPoleParams = DEFAULT_PARAMS):
    theta0 = 0.2
    state0 = np.array([0.0, theta0, 0.0, 0.0])
    T = 5.0
    dt = 0.02
    Q = np.diag([1.0, 10.0, 1.0, 10.0])
    R_val = 1.0
    R = np.array([[R_val]])

    # --- PID ---
    Kp, Ki, Kd = 50.0, 0.0, 20.0
    integral = [0.0]
    prev_err = [0.0]

    def pid_ctrl(t, state):
        e = -state[1]
        integral[0] += e * dt
        integral[0] = np.clip(integral[0], -5, 5)
        de = (e - prev_err[0]) / dt
        prev_err[0] = e
        return Kp * e + Ki * integral[0] + Kd * de

    ts_pid, st_pid, u_pid = simulate_cartpole(state0, pid_ctrl, T, dt, p)

    # --- LQR ---
    A, B = linearize_cartpole(p)
    K, _ = solve_lqr(A, B, Q, R)

    def lqr_ctrl(t, state):
        return float(-K @ state)

    ts_lqr, st_lqr, u_lqr = simulate_cartpole(state0, lqr_ctrl, T, dt, p)

    def cost(states, controls):
        c = 0.0
        for i in range(len(controls)):
            x = states[i]
            c += (x @ Q @ x + R_val * controls[i] ** 2) * dt
        return c

    J_pid = cost(st_pid, u_pid)
    J_lqr = cost(st_lqr, u_lqr)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(ts_pid, st_pid[:, 0], label="x")
    axes[0, 0].plot(ts_pid, st_pid[:, 1], label="θ")
    axes[0, 0].axhline(0, color="gray", ls="--", lw=0.5)
    axes[0, 0].set_title(f"PID States  (J={J_pid:.2f})")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("State")

    axes[0, 1].plot(ts_lqr, st_lqr[:, 0], label="x")
    axes[0, 1].plot(ts_lqr, st_lqr[:, 1], label="θ")
    axes[0, 1].axhline(0, color="gray", ls="--", lw=0.5)
    axes[0, 1].set_title(f"LQR States  (J={J_lqr:.2f})")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ts_pid, u_pid, "g", lw=1.5)
    axes[1, 0].set_title("PID Control")
    axes[1, 0].set_xlabel("t [s]")
    axes[1, 0].set_ylabel("u [N]")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ts_lqr, u_lqr, "g", lw=1.5)
    axes[1, 1].set_title("LQR Control")
    axes[1, 1].set_xlabel("t [s]")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("PID vs LQR on CartPole Balancing", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show()

    print(f"Quadratic cost J = xᵀQx + uᵀRu:")
    print(f"  PID: J = {J_pid:.3f}")
    print(f"  LQR: J = {J_lqr:.3f}  (optimal for this Q, R)")

    return animate_cartpole(ts_lqr, st_lqr, u_lqr, p)
