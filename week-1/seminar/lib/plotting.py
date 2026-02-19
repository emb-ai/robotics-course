from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_frame_3d(
    R: np.ndarray,
    origin: None | np.ndarray = None,
    scale: float = 1.0,
    ax=None,
    label: str = "",
    colors: list[str] | None = None,
) -> plt.Axes:
    if origin is None:
        origin = np.zeros(3)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]
    for i in range(3):
        direction = R[:, i] * scale
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=colors[i],
            arrow_length_ratio=0.2,
            label=f"{label}{labels[i]}" if label else labels[i],
        )
    ax.set_box_aspect((1, 1, 1))
    return ax


def plot_planar_robot(
    theta: np.ndarray,
    L: None | list[float] = None,
    ax=None,
) -> plt.Axes:
    if L is None:
        L = [1.0, 1.0, 0.5]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.atleast_1d(theta)
    L = L[: len(theta)]
    x = [0.0]
    y = [0.0]
    angle_sum = 0.0
    for t, l in zip(theta, L):
        angle_sum += t
        x.append(x[-1] + l * np.cos(angle_sum))
        y.append(y[-1] + l * np.sin(angle_sum))
    ax.plot(x, y, "o-", linewidth=3, markersize=8, label="Joints")
    ax.plot(x[0], y[0], "ks", markersize=12, label="Base")
    ax.plot(x[-1], y[-1], "r*", markersize=15, label="End-effector")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax
