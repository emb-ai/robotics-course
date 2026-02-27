from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ipywidgets import interact, FloatSlider


# ---------------------------------------------------------------------------
# Phase portrait
# ---------------------------------------------------------------------------

def show_phase_portrait(
    A: NDArray,
    title: str | None = None,
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    eigvals = np.linalg.eigvals(A)
    eig_str = ", ".join(f"{v:.2g}" for v in eigvals)
    default_title = f"λ = {eig_str}"
    resolved_title = title if title is not None else default_title

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))

    xs = np.linspace(xlim[0], xlim[1], 25)
    ys = np.linspace(ylim[0], ylim[1], 25)
    X, Y = np.meshgrid(xs, ys)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    speed = np.sqrt(U**2 + V**2)
    speed_norm = speed / (speed.max() + 1e-12)

    ax.streamplot(X, Y, U, V, color=speed_norm, cmap="coolwarm", density=1.4, linewidth=0.8)
    ax.plot(0, 0, "ko", markersize=6, zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(resolved_title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Gallery of canonical phase portraits
# ---------------------------------------------------------------------------

_STABILITY_EXAMPLES: list[tuple[str, NDArray]] = [
    ("Stable node",     np.array([[-2, 0], [0, -1]], dtype=float)),
    ("Stable spiral",   np.array([[-0.5, 2], [-2, -0.5]], dtype=float)),
    ("Center",          np.array([[0, 1], [-1, 0]], dtype=float)),
    ("Unstable node",   np.array([[1, 0], [0, 2]], dtype=float)),
    ("Unstable spiral", np.array([[0.5, 2], [-2, 0.5]], dtype=float)),
    ("Saddle",          np.array([[-1, 0], [0, 1]], dtype=float)),
]


def show_stability_examples() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (label, A) in zip(axes.flat, _STABILITY_EXAMPLES):
        eigvals = np.linalg.eigvals(A)
        eig_str = ", ".join(f"{v:.2g}" for v in eigvals)
        title = f"{label}  (λ = {eig_str})"
        show_phase_portrait(A, title=title, ax=ax)
    fig.suptitle("Canonical 2D Linear Phase Portraits", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interactive eigenvalue / response explorer
# ---------------------------------------------------------------------------

def show_eigenvalue_response_interactive() -> None:
    fig, (ax_phase, ax_resp) = plt.subplots(1, 2, figsize=(11, 4.5))

    def _update(a11: float = -1.0, a12: float = 0.0, a21: float = 0.0, a22: float = -1.0):
        A = np.array([[a11, a12], [a21, a22]])

        ax_phase.clear()
        show_phase_portrait(A, ax=ax_phase)

        ax_resp.clear()
        dt = 0.01
        T = 6.0
        ts = np.arange(0, T, dt)
        x = np.array([1.0, 0.0])
        traj = np.zeros((len(ts), 2))
        for i, t in enumerate(ts):
            traj[i] = x
            x = x + dt * (A @ x)

        ax_resp.plot(ts, traj[:, 0], label="$x_1(t)$")
        ax_resp.plot(ts, traj[:, 1], label="$x_2(t)$")
        ax_resp.axhline(0, color="gray", lw=0.5)
        ax_resp.set_xlabel("t [s]")
        ax_resp.set_ylabel("state")
        ax_resp.set_title("Step response from $x_0=[1,0]$", fontsize=10)
        ax_resp.legend(fontsize=8)
        ax_resp.grid(True, alpha=0.3)
        y_max = max(np.abs(traj).max(), 0.5)
        ax_resp.set_ylim(-min(y_max, 10), min(y_max, 10))

        fig.tight_layout()
        fig.canvas.draw_idle()

    slider_kw = dict(min=-3.0, max=3.0, step=0.1)
    interact(
        _update,
        a11=FloatSlider(value=-1.0, description="a₁₁", **slider_kw),
        a12=FloatSlider(value=0.0,  description="a₁₂", **slider_kw),
        a21=FloatSlider(value=0.0,  description="a₂₁", **slider_kw),
        a22=FloatSlider(value=-1.0, description="a₂₂", **slider_kw),
    )
    plt.show()


# ---------------------------------------------------------------------------
# Controllability demo
# ---------------------------------------------------------------------------

def show_controllability_demo() -> None:
    examples: list[tuple[str, NDArray, NDArray]] = [
        ("Controllable",     np.array([[0, 1], [0, 0]]),  np.array([[0], [1]])),
        ("Not controllable", np.array([[1, 0], [0, 2]]),  np.array([[0], [0]])),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (label, A, B) in zip(axes, examples):
        C = np.hstack([B, A @ B])
        rank = np.linalg.matrix_rank(C)
        n = A.shape[0]
        controllable = rank == n

        ax.set_title(f"{label}  (rank C = {rank}/{n})", fontsize=11, fontweight="bold")

        if controllable:
            dt = 0.05
            n_dirs = 60
            T_reach = 1.0
            points = []
            for sign in [1.0, -1.0]:
                for angle_idx in range(n_dirs):
                    theta = 2 * np.pi * angle_idx / n_dirs
                    x = np.array([0.0, 0.0])
                    for _ in np.arange(0, T_reach, dt):
                        u = sign * np.array([np.sin(theta * _ / T_reach)])
                        x = x + dt * (A @ x + (B @ u).ravel())
                    points.append(x)
            points_arr = np.array(points)
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_arr)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.fill(points_arr[hull_pts, 0], points_arr[hull_pts, 1],
                    alpha=0.25, color="#55A868", label="Reachable set (approx)")
            ax.plot(points_arr[:, 0], points_arr[:, 1], ".", color="#55A868", markersize=2)
        else:
            ax.text(0, 0, "Reachable set = {0}\n(no control authority)",
                    ha="center", va="center", fontsize=10, color="#C44E52",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#C44E52"))

        ax.plot(0, 0, "ko", markersize=6, zorder=5)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

        cmat_str = f"C = [B | AB] =\n{C}"
        ax.text(0.98, 0.02, cmat_str, transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Controllability: Rank of [B | AB | … | Aⁿ⁻¹B]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Open-loop vs closed-loop
# ---------------------------------------------------------------------------

def show_open_vs_closed_loop() -> None:
    a = 1.0   # unstable pole
    b = 1.0
    k = -3.0  # feedback gain → closed-loop pole at a + b*k = -2
    dt = 0.01
    T = 4.0
    ts = np.arange(0, T, dt)
    x0 = 0.5

    x_open = np.zeros_like(ts)
    x_closed = np.zeros_like(ts)
    u_closed = np.zeros_like(ts)
    x_open[0] = x_closed[0] = x0

    for i in range(len(ts) - 1):
        x_open[i + 1] = x_open[i] + dt * a * x_open[i]

        u = k * x_closed[i]
        u_closed[i] = u
        x_closed[i + 1] = x_closed[i] + dt * (a * x_closed[i] + b * u)

    fig, (ax_ol, ax_cl) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    ax_ol.plot(ts, x_open, color="#C44E52", linewidth=2)
    ax_ol.axhline(0, color="gray", lw=0.5)
    ax_ol.set_title(f"Open-loop  (a={a})", fontsize=11, fontweight="bold")
    ax_ol.set_xlabel("t [s]")
    ax_ol.set_ylabel("x(t)")
    ax_ol.set_ylim(-1, min(x_open.max() * 1.1, 50))
    ax_ol.grid(True, alpha=0.3)
    ax_ol.annotate("Diverges!", xy=(ts[-1], x_open[-1]),
                   fontsize=10, color="#C44E52", fontweight="bold",
                   ha="right", va="bottom")

    ax_cl.plot(ts, x_closed, color="#4C72B0", linewidth=2, label="x(t)")
    ax_cl.plot(ts, u_closed, color="#55A868", linewidth=1.5, linestyle="--", label="u(t)=kx", alpha=0.7)
    ax_cl.axhline(0, color="gray", lw=0.5)
    ax_cl.set_title(f"Closed-loop  (a+bk={a + b * k:.1f})", fontsize=11, fontweight="bold")
    ax_cl.set_xlabel("t [s]")
    ax_cl.set_ylabel("x(t)")
    ax_cl.legend(fontsize=9)
    ax_cl.grid(True, alpha=0.3)
    ax_cl.annotate("Stabilized", xy=(ts[len(ts) // 2], 0.02),
                   fontsize=10, color="#4C72B0", fontweight="bold")

    fig.suptitle("Why Feedback Matters: Unstable 1D System  ẋ = ax + bu", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()
