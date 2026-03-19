"""Visualization helpers for the week-6 lecture notebook."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def show_image_grid(
    images: Sequence[NDArray],
    titles: Sequence[str] | None = None,
    cols: int = 3,
    figsize_per_img: tuple[float, float] = (3.0, 2.5),
) -> plt.Figure:
    """Display a grid of RGB images."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_img[0] * cols, figsize_per_img[1] * rows))
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < n:
            ax.imshow(images[idx])
            if titles:
                ax.set_title(titles[idx], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_reward_landscape_2d(
    reward_fn,
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
    resolution: int = 200,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Reward landscape",
) -> plt.Figure:
    """Contour plot of a 2-input reward function."""
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda x, y: reward_fn(x, y))(X, Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="reward")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_trajectories(
    trajectories: dict[str, NDArray],
    goal: NDArray | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Trajectory comparison",
) -> plt.Figure:
    """Plot named 2D trajectories (N, 2) and optional goal point."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, traj in trajectories.items():
        ax.plot(traj[:, 0], traj[:, 1], label=name, linewidth=1.5)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=40, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=60, zorder=5)
    if goal is not None:
        ax.scatter(*goal, marker="*", s=200, c="gold", edgecolors="k", zorder=6, label="goal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_reward_curves(
    curves: dict[str, NDArray],
    xlabel: str = "Episode",
    ylabel: str = "Return",
    title: str = "Training curves",
) -> plt.Figure:
    """Plot reward / return curves for multiple runs."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in curves.items():
        ax.plot(vals, label=name, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_covariate_shift_demo(
    expert_states: NDArray,
    bc_states: NDArray,
    dagger_states: NDArray | None = None,
    title: str = "Covariate shift: BC vs Expert",
) -> plt.Figure:
    """Scatter plot showing state distributions visited by expert vs. learned policies."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(expert_states[:, 0], expert_states[:, 1],
               alpha=0.4, s=15, label="Expert", c="tab:green")
    ax.scatter(bc_states[:, 0], bc_states[:, 1],
               alpha=0.4, s=15, label="BC policy", c="tab:red")
    if dagger_states is not None:
        ax.scatter(dagger_states[:, 0], dagger_states[:, 1],
                   alpha=0.4, s=15, label="DAgger policy", c="tab:blue")
    ax.set_xlabel("State dim 1")
    ax.set_ylabel("State dim 2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
