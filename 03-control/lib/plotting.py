from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_episode(
    ts: NDArray,
    states: NDArray,
    controls: NDArray,
    rewards: NDArray | None = None,
    title: str = "",
    target_x: float = 0.0,
):
    n_plots = 4 if rewards is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)

    axes[0].plot(ts, states[:, 0], label="x")
    axes[0].plot(ts, states[:, 1], label=r"$\theta$")
    if target_x != 0.0:
        axes[0].axhline(target_x, color="red", ls="--", lw=1, alpha=0.6, label=f"target x={target_x:.1f}")
    axes[0].set_ylabel("position / angle")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, states[:, 2], label=r"$\dot{x}$")
    axes[1].plot(ts, states[:, 3], label=r"$\dot{\theta}$")
    axes[1].set_ylabel("velocity")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts, controls, color="green")
    axes[2].set_ylabel("u [N]")
    axes[2].grid(True, alpha=0.3)

    if rewards is not None:
        axes[3].plot(ts, rewards, color="purple")
        axes[3].set_ylabel("reward")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlabel("t [s]")
    else:
        axes[2].set_xlabel("t [s]")

    if title:
        axes[0].set_title(title)

    fig.tight_layout()
    return fig, axes
