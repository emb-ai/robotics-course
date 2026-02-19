from __future__ import annotations

import numpy as np


def set_3d_equal_aspect(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, margin: float = 0.1) -> None:
    cx = (np.min(xs) + np.max(xs)) / 2
    cy = (np.min(ys) + np.max(ys)) / 2
    cz = (np.min(zs) + np.max(zs)) / 2
    half = max(np.ptp(xs), np.ptp(ys), np.ptp(zs)) / 2
    half = max(half, 1e-6) * (1 + margin)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect((1, 1, 1))
