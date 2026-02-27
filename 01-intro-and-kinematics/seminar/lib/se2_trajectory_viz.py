from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.ndimage import rotate as ndimage_rotate

from .rotations import rot2d
from .se2 import se2_exp

_THETA_LO, _THETA_HI = -np.pi, np.pi
_N_SAMPLES = 600
_SPRITE_SAMPLE_EVERY = 10
_SPRITE_ZOOM = 0.15
_PLANE_ALPHA = 0.35
_PLANE_EXTENT = 4.0

_INITIAL_X = -0.8
_INITIAL_Y = 0.4
_INITIAL_THETA = -np.pi + 0.08


def _se2_from_xy_theta(x: float, y: float, theta: float) -> np.ndarray:
    R = rot2d(theta)
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = np.array([x, y])
    return T


def _initial_pose() -> np.ndarray:
    return _se2_from_xy_theta(_INITIAL_X, _INITIAL_Y, _INITIAL_THETA)


def _twist_sequence_drift() -> list[tuple[float, float, np.ndarray]]:
    return [
        (0.5, 4.0 * np.pi, np.array([1.0, 0.0])),
        (0.35, -2.5, np.array([0.9, 0.0])),
        (0.4, 1.8, np.array([1.0, 0.0])),
        (0.3, -1.2, np.array([0.7, 0.0])),
        (0.35, 0.0, np.array([0.6, 0.0])),
    ]


def _trajectory_poses(
    segments: list[tuple[float, float, np.ndarray]],
    n_samples: int = _N_SAMPLES,
    T_init: np.ndarray | None = None,
) -> list[np.ndarray]:
    if T_init is None:
        T_init = _initial_pose()
    total_duration = sum(s[0] for s in segments)
    taus = np.linspace(0, total_duration, n_samples, endpoint=True)
    poses: list[np.ndarray] = []
    T = T_init.copy()
    tau_prev = 0.0
    seg_idx = 0
    omega, v = segments[0][1], segments[0][2]
    delta_tau = segments[0][0]
    for tau in taus:
        while seg_idx < len(segments) and tau > tau_prev + delta_tau:
            T = T @ se2_exp(omega, v, 1.0)
            seg_idx += 1
            tau_prev += delta_tau
            if seg_idx < len(segments):
                delta_tau, omega, v = segments[seg_idx][0], segments[seg_idx][1], segments[seg_idx][2]
        if seg_idx >= len(segments):
            poses.append(T.copy())
            continue
        s = (tau - tau_prev) / delta_tau
        s = np.clip(s, 0.0, 1.0)
        T_at_tau = T @ se2_exp(omega, v, s)
        poses.append(T_at_tau.copy())
    return poses


def _T_to_xy_theta(T: np.ndarray) -> tuple[float, float, float]:
    x, y = T[0, 2], T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta


def _draw_se2_slab(ax: Axes3D.Axes) -> None:
    extent = _PLANE_EXTENT
    n = 12
    u = np.linspace(-extent, extent, n)
    v = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(u, v)
    for z_val, label in [(_THETA_LO, r"$\theta = -\pi$"), (_THETA_HI, r"$\theta = \pi$")]:
        Z = np.full_like(X, z_val)
        ax.plot_surface(X, Y, Z, alpha=_PLANE_ALPHA, color="pink", edgecolor="red", linewidth=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel(r"$\theta$")
    ax.set_zlim(_THETA_LO, _THETA_HI)


def _draw_trajectory_3d(ax: Axes3D.Axes, poses: list[np.ndarray]) -> None:
    xs = []
    ys = []
    thetas = []
    for T in poses:
        x, y, theta = _T_to_xy_theta(T)
        xs.append(x)
        ys.append(y)
        thetas.append(np.arctan2(np.sin(theta), np.cos(theta)))
    ax.plot(xs, ys, thetas, color="steelblue", linewidth=0.5, marker='o', markersize=4, alpha=0.85)


def _draw_2d_trajectory_with_sprites(
    ax: plt.Axes,
    poses: list[np.ndarray],
    sprite_path: Path,
) -> None:
    all_x = []
    all_y = []
    for T in poses:
        x, y, _ = _T_to_xy_theta(T)
        all_x.append(x)
        all_y.append(y)
    ax.plot(all_x, all_y, color="gray", linewidth=1, alpha=0.8, zorder=0)
    try:
        img = plt.imread(str(sprite_path))
    except Exception:
        img = None
    if img is not None:
        indices = sorted(set(range(0, len(poses), _SPRITE_SAMPLE_EVERY)) | {len(poses) - 1})
        sampled = [poses[i] for i in indices]
        for T in sampled:
            x, y, theta = _T_to_xy_theta(T)
            deg = np.degrees(theta)
            img_rot = ndimage_rotate(img, -deg, reshape=True, order=1, mode="constant", cval=0)
            if img_rot.ndim == 2:
                img_rot = np.stack([img_rot] * 3 + [np.ones_like(img_rot)], axis=-1)
            elif img_rot.shape[2] == 4:
                pass
            else:
                img_rot = np.dstack([img_rot, np.ones(img_rot.shape[:2])])
            imagebox = OffsetImage(img_rot, zoom=_SPRITE_ZOOM)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = x_max - x_min
    y_span = y_max - y_min
    pad = max(0.02 * max(x_span, y_span) if (x_span or y_span) else 0.1, 0.05)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True, alpha=0.3)


def show_se2_drift_trajectory(
    figsize: tuple[float, float] = (12, 5),
    sprite_path: Path | None = None,
) -> None:
    if sprite_path is None:
        sprite_path = Path(__file__).resolve().parent.parent / "assets" / "car_sprite.png"
    segments = _twist_sequence_drift()
    poses = _trajectory_poses(segments)
    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(121, projection="3d")
    _draw_se2_slab(ax_3d)
    _draw_trajectory_3d(ax_3d, poses)
    ax_3d.set_title(r"""SE(2) $\simeq \mathbb{R}^2 \times S^1$
    (red planes are glued together with the same orientation)""")
    ax_2d = fig.add_subplot(122)
    _draw_2d_trajectory_with_sprites(ax_2d, poses, sprite_path)
    ax_2d.set_title("Trajectory (body twist drift)")
    plt.tight_layout()
    plt.show()
