from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Circle

_TORUS_R, _TORUS_r = 1.2, 0.4


def plot_annulus_from_circles(
    ax: plt.Axes,
    L1: float,
    L2: float,
    n_circles: int = 20,
    n_points: int = 80,
    cmap: str | None = "hsv",
) -> None:
    R_inner = abs(L1 - L2)
    R_outer = L1 + L2
    outer = Circle(
        (0, 0), R_outer, fill=True, facecolor="lightgray", edgecolor="none", alpha=0.4, zorder=0
    )
    ax.add_patch(outer)
    if R_inner > 1e-6:
        inner = Circle(
            (0, 0), R_inner, fill=True, facecolor="white", edgecolor="none", zorder=1
        )
        ax.add_patch(inner)
    theta1_values = np.linspace(0, 2 * np.pi, n_circles, endpoint=False)
    theta2 = np.linspace(0, 2 * np.pi, n_points)
    colors = get_cmap(cmap)(np.linspace(0, 1, n_circles, endpoint=False)) if cmap else None
    for i, t1 in enumerate(theta1_values):
        x = L1 * np.cos(t1) + L2 * np.cos(t1 + theta2)
        y = L1 * np.sin(t1) + L2 * np.sin(t1 + theta2)
        color = colors[i] if colors is not None else "steelblue"
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.8, zorder=2)
    ax.set_xlim(-R_outer * 1.15, R_outer * 1.15)
    ax.set_ylim(-R_outer * 1.15, R_outer * 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Workspace")
    ax.grid(True, alpha=0.3)


def plot_torus_from_circles(
    ax: plt.Axes,
    R: float = _TORUS_R,
    r: float = _TORUS_r,
    n_circles: int = 20,
    n_points: int = 80,
    cmap: str | None = "hsv",
) -> None:
    theta1_values = np.linspace(0, 2 * np.pi, n_circles, endpoint=False)
    theta2 = np.linspace(0, 2 * np.pi, n_points)
    colors = get_cmap(cmap)(np.linspace(0, 1, n_circles, endpoint=False)) if cmap else None
    for i, t1 in enumerate(theta1_values):
        x = (R + r * np.cos(theta2)) * np.cos(t1)
        y = (R + r * np.cos(theta2)) * np.sin(t1)
        z = r * np.sin(theta2)
        color = colors[i] if colors is not None else "steelblue"
        ax.plot(x, y, z, color=color, linewidth=0.8, alpha=0.8)
    lim = R + r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title("Configuration space")
    ax.set_box_aspect((1, 1, 1))


def plot_torus_workspace_mapping(
    L1: float = 1.0,
    L2: float = 0.5,
    n_circles: int = 20,
    n_points: int = 80,
    torus_R: float = _TORUS_R,
    torus_r: float = _TORUS_r,
    figsize: tuple[float, float] = (10, 5),
    cmap: str = "hsv",
) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    ax_annulus = fig.add_subplot(121)
    ax_torus = fig.add_subplot(122, projection="3d")
    plot_annulus_from_circles(
        ax_annulus, L1, L2, n_circles=n_circles, n_points=n_points, cmap=cmap
    )
    plot_torus_from_circles(
        ax_torus, R=torus_R, r=torus_r, n_circles=n_circles, n_points=n_points, cmap=cmap
    )
    plt.tight_layout()
