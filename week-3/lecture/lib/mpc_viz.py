from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg

from .cartpole_sim import (
    DEFAULT_PARAMS, CartPoleParams,
    linearize_cartpole, cartpole_dynamics,
    animate_cartpole,
)


def discretize_linear(A: NDArray, B: NDArray, dt: float) -> tuple[NDArray, NDArray]:
    n = A.shape[0]
    em = scipy.linalg.expm(A * dt)
    try:
        A_inv = np.linalg.inv(A)
        Bd = A_inv @ (em - np.eye(n)) @ B
    except np.linalg.LinAlgError:
        Bd = B * dt
    return em, Bd


def solve_mpc(
    x0: NDArray, Ad: NDArray, Bd: NDArray,
    N: int, Q: NDArray, R: NDArray, Qf: NDArray,
    u_min: float, u_max: float,
) -> tuple[NDArray, NDArray]:
    nx = Ad.shape[0]
    nu = Bd.shape[1]

    def cost_and_states(u_flat):
        U = u_flat.reshape(N, nu)
        x = x0.copy()
        J = 0.0
        xs = [x.copy()]
        for k in range(N):
            J += float(x @ Q @ x + U[k] @ R @ U[k])
            x = Ad @ x + Bd @ U[k]
            xs.append(x.copy())
        J += float(x @ Qf @ x)
        return J, np.array(xs)

    def cost(u_flat):
        return cost_and_states(u_flat)[0]

    def grad(u_flat):
        U = u_flat.reshape(N, nu)
        xs = [x0.copy()]
        x = x0.copy()
        for k in range(N):
            x = Ad @ x + Bd @ U[k]
            xs.append(x.copy())

        lambdas = [2 * Qf @ xs[-1]]
        for k in range(N - 1, -1, -1):
            lam = 2 * Q @ xs[k] + Ad.T @ lambdas[-1]
            lambdas.append(lam)
        lambdas.reverse()

        g = np.zeros_like(u_flat).reshape(N, nu)
        for k in range(N):
            g[k] = 2 * R @ U[k] + Bd.T @ lambdas[k + 1]
        return g.ravel()

    u0 = np.zeros(N * nu)
    bounds = [(u_min, u_max)] * (N * nu)
    res = scipy.optimize.minimize(cost, u0, jac=grad, bounds=bounds, method="L-BFGS-B",
                                  options={"maxiter": 200, "ftol": 1e-10})
    _, xs = cost_and_states(res.x)
    return res.x.reshape(N, nu), xs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_mpc_concept():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 5)
    ax.axis("off")

    # Past trajectory
    past_t = np.array([0, 1, 2, 3, 4])
    past_y = np.array([3.5, 3.0, 2.2, 1.8, 1.5])
    ax.plot(past_t, past_y, "b-o", lw=2, markersize=6, label="Past trajectory")

    # Current state
    ax.plot(4, 1.5, "ro", markersize=12, zorder=10, label="Current state")

    # Predicted trajectory
    pred_t = np.array([4, 5, 6, 7, 8, 9])
    pred_y = np.array([1.5, 1.1, 0.7, 0.4, 0.2, 0.1])
    ax.plot(pred_t, pred_y, "b--o", lw=1.5, markersize=5, alpha=0.6, label="Predicted trajectory")

    # Control bars
    bar_w = 0.6
    u_vals = [0.8, 0.6, 0.4, 0.3, 0.2]
    for i, u in enumerate(u_vals):
        color = "green" if i == 0 else "lightgreen"
        edge = "darkgreen" if i == 0 else "green"
        ax.bar(pred_t[i] + 0.5, u, width=bar_w, bottom=-0.8, color=color,
               edgecolor=edge, alpha=0.8 if i == 0 else 0.4)

    ax.annotate("Apply only u₀", xy=(4.5, 0.05), fontsize=10, color="darkgreen", fontweight="bold")
    ax.annotate("", xy=(4.5, -0.15), xytext=(4.5, -0.55),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))

    # Horizon bracket
    ax.annotate("", xy=(9, 4.2), xytext=(4, 4.2),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax.text(6.5, 4.4, "Prediction horizon N", ha="center", fontsize=10, color="purple")

    # Next step
    ax.annotate("Shift & re-solve", xy=(5, 1.1), xytext=(7, 3.5),
                fontsize=10, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("Model Predictive Control: Receding Horizon", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plt.show()


def show_mpc_cartpole(
    N: int = 20,
    Q_diag: tuple[float, ...] = (1, 10, 1, 10),
    R_val: float = 0.1,
    theta0: float = 0.3,
    f_max: float = 10.0,
    T: float = 3.0,
    dt: float = 0.02,
    p: CartPoleParams = DEFAULT_PARAMS,
):
    A, B = linearize_cartpole(p)
    Ad, Bd = discretize_linear(A, B, dt)
    Q = np.diag(Q_diag)
    R_mat = np.array([[R_val]])
    Qf = 10 * Q

    n_steps = int(T / dt)
    state = np.array([0.0, theta0, 0.0, 0.0])

    ts = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 4))
    controls = np.zeros(n_steps + 1)
    states[0] = state

    pred_snapshots = []

    for i in range(n_steps):
        U_opt, X_pred = solve_mpc(state, Ad, Bd, N, Q, R_mat, Qf, -f_max, f_max)
        u = float(U_opt[0, 0])
        controls[i] = u

        if i % 10 == 0:
            pred_ts = ts[i] + np.arange(N + 1) * dt
            pred_snapshots.append((pred_ts, X_pred.copy()))

        deriv = cartpole_dynamics(state, u, p)
        state = state + dt * deriv
        states[i + 1] = state
        ts[i + 1] = (i + 1) * dt

    controls[-1] = controls[-2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ts, states[:, 0], label="x [m]")
    ax.plot(ts, states[:, 1], label="θ [rad]")
    for pt, px in pred_snapshots:
        ax.plot(pt, px[:, 1], "r--", alpha=0.25, lw=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("State")
    ax.set_title("MPC States + Predicted Trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts, controls, "g", lw=1.5)
    ax.axhline(f_max, color="r", ls=":", alpha=0.5, label=f"±{f_max} N")
    ax.axhline(-f_max, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("u [N]")
    ax.set_title("MPC Control")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(states[:, 1], states[:, 3], "b", lw=1)
    ax.plot(states[0, 1], states[0, 3], "go", markersize=8, label="start")
    ax.plot(0, 0, "r*", markersize=12, label="target")
    ax.set_xlabel("θ [rad]")
    ax.set_ylabel("θ̇ [rad/s]")
    ax.set_title("Phase Portrait")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"MPC on CartPole (N={N}, f_max={f_max})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

    return animate_cartpole(ts, states, controls, p)


def show_mpc_constrained_vs_unconstrained(p: CartPoleParams = DEFAULT_PARAMS):
    A, B = linearize_cartpole(p)
    dt = 0.02
    Ad, Bd = discretize_linear(A, B, dt)
    Q = np.diag([1.0, 10.0, 1.0, 10.0])
    R_mat = np.array([[0.1]])
    Qf = 10 * Q
    N = 20
    theta0 = 0.4
    T = 4.0
    n_steps = int(T / dt)

    results = {}
    for label, f_max in [("Unconstrained (f_max=100)", 100.0), ("Constrained (f_max=3)", 3.0)]:
        state = np.array([0.0, theta0, 0.0, 0.0])
        ts = np.zeros(n_steps + 1)
        states = np.zeros((n_steps + 1, 4))
        controls = np.zeros(n_steps + 1)
        states[0] = state

        for i in range(n_steps):
            U_opt, _ = solve_mpc(state, Ad, Bd, N, Q, R_mat, Qf, -f_max, f_max)
            u = float(U_opt[0, 0])
            controls[i] = u
            deriv = cartpole_dynamics(state, u, p)
            state = state + dt * deriv
            states[i + 1] = state
            ts[i + 1] = (i + 1) * dt

        controls[-1] = controls[-2]
        results[label] = (ts, states, controls, f_max)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, (label, (ts, states, controls, f_max)) in enumerate(results.items()):
        axes[0, col].plot(ts, states[:, 0], label="x")
        axes[0, col].plot(ts, states[:, 1], label="θ")
        axes[0, col].axhline(0, color="gray", ls="--", lw=0.5)
        axes[0, col].set_title(label)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel("State")

        axes[1, col].plot(ts, controls, "g", lw=1.5)
        axes[1, col].axhline(f_max, color="r", ls=":", alpha=0.5)
        axes[1, col].axhline(-f_max, color="r", ls=":", alpha=0.5)
        axes[1, col].set_xlabel("t [s]")
        axes[1, col].set_title(f"Control (±{f_max} N)")
        axes[1, col].grid(True, alpha=0.3)
        if col == 0:
            axes[1, col].set_ylabel("u [N]")

    fig.suptitle("MPC: Effect of Control Constraints", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show()
