from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, FloatSlider

from .contact_friction import simulate_bouncing_ball_penalty, simulate_bouncing_ball_impulse


def show_bouncing_ball_comparison(
    y0: float = 2.0, vy0: float = 0.0, dt: float = 0.0005,
    n_steps: int = 20000, restitution: float = 0.8,
    k: float = 1e4, c: float = 100.0,
):
    ts_p, ys_p, vs_p = simulate_bouncing_ball_penalty(
        y0, vy0, dt, n_steps, k=k, c=c,
    )
    ts_i, ys_i, vs_i = simulate_bouncing_ball_impulse(
        y0, vy0, dt, n_steps, restitution=restitution,
    )

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts_p, ys_p, color="#e74c3c", lw=1.0, label="Penalty")
    ax1.plot(ts_i, ys_i, color="#3498db", lw=1.0, label="Impulse")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel("y")
    ax1.set_title("Height y(t)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts_p, vs_p, color="#e74c3c", lw=0.8, label="Penalty")
    ax2.plot(ts_i, vs_i, color="#3498db", lw=0.8, label="Impulse")
    ax2.set_xlabel("t")
    ax2.set_ylabel("vy")
    ax2.set_title("Velocity vy(t)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # zoom into penetration
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ts_p, ys_p, color="#e74c3c", lw=1.0, label="Penalty")
    ax3.axhline(0, color="k", lw=0.5)
    min_y = np.min(ys_p)
    if min_y < 0:
        ax3.set_ylim(min_y * 1.5, max(0.1, -min_y * 5))
    else:
        ax3.set_ylim(-0.01, 0.2)
    ax3.set_xlabel("t")
    ax3.set_ylabel("y")
    ax3.set_title(f"Penalty method: penetration (min y = {min_y:.4f})")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # energy
    ax4 = fig.add_subplot(gs[1, 1])
    g = 9.81
    m = 1.0
    E_p = 0.5 * m * vs_p**2 + m * g * ys_p
    E_i = 0.5 * m * vs_i**2 + m * g * np.maximum(ys_i, 0)
    ax4.plot(ts_p, E_p, color="#e74c3c", lw=1.0, label="Penalty")
    ax4.plot(ts_i, E_i, color="#3498db", lw=1.0, label="Impulse")
    ax4.set_xlabel("t")
    ax4.set_ylabel("E")
    ax4.set_title("Mechanical energy")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_restitution_comparison(
    y0: float = 2.0, dt: float = 0.0005, n_steps: int = 20000,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, e in zip(axes, [0.0, 0.5, 1.0]):
        ts, ys, vs = simulate_bouncing_ball_impulse(
            y0, 0.0, dt, n_steps, restitution=e,
        )
        ax.plot(ts, ys, "k-", lw=1.0)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title(f"e = {e}")
        ax.set_ylim(-0.1, y0 * 1.2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Impulse-based bouncing ball: varying restitution coefficient", y=1.02)
    fig.tight_layout()
    plt.show()


def show_bouncing_ball_interactive(y0: float = 2.0):
    @interact(
        restitution=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.8, description="e"),
        log_k=FloatSlider(min=2.0, max=6.0, step=0.5, value=4.0, description="log₁₀(k)"),
    )
    def _update(restitution, log_k):
        show_bouncing_ball_comparison(
            y0=y0, restitution=restitution, k=10**log_k,
        )
