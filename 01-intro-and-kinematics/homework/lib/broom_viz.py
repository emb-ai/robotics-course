from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lib.broom_types import Configuration, XYZConfiguration, KAPPA_MAX, PHI_MIN, PHI_MAX, V


def show_broom_path(
    curve_fn: Callable[[np.ndarray], Configuration],
    start: Configuration,
    goal: Configuration | XYZConfiguration | None = None,
    n_points: int = 200,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    s = np.linspace(0.0, 1.0, n_points)
    xs, ys, zs, phis = [], [], [], []
    for si in s:
        c = curve_fn(np.atleast_1d(si))
        xs.append(c.x)
        ys.append(c.y)
        zs.append(c.z)
        phis.append(c.phi)
    xs, ys, zs, phis = np.array(xs), np.array(ys), np.array(zs), np.array(phis)

    pitch_ok = (phis >= PHI_MIN - 1e-3) & (phis <= PHI_MAX + 1e-3)

    created_fig = ax is None
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    for i in range(len(xs) - 1):
        color = "tab:green" if pitch_ok[i] else "tab:red"
        ax.plot(xs[i : i + 2], ys[i : i + 2], zs[i : i + 2], color=color, linewidth=1.5)

    ax.scatter(*start.position(), color="blue", s=60, marker="o", label="start")
    scale = 0.3
    ax.quiver(*start.position(), *start.direction() * scale, color="blue", arrow_length_ratio=0.2)

    if goal is not None:
        gp = goal.position()
        ax.scatter(*gp, color="red", s=60, marker="x", label="goal")
        if isinstance(goal, Configuration):
            ax.quiver(*gp, *goal.direction() * scale, color="red", arrow_length_ratio=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    if created_fig:
        plt.tight_layout()
    return fig if created_fig else None
