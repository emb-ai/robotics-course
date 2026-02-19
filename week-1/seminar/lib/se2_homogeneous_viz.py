from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .rotations import rot2d

_PLANE_EXTENT = 2.5
_FRAME_SCALE = 1.0
_GRID_STEP = 1.0
_GRID_RANGE = 2.0
_GRID_W_VALS = np.array([0.0, 1.0, 2.0])
_AXIS_SCALE = 1.2


def _se2_matrix(x: float, y: float, theta: float) -> np.ndarray:
    R = rot2d(theta)
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = np.array([x, y])
    return T


def _reference_frame_points() -> np.ndarray:
    origin = np.array([[0], [0], [1]])
    x_axis = np.array([[_FRAME_SCALE], [0], [1]])
    y_axis = np.array([[0], [_FRAME_SCALE], [1]])
    return np.hstack([origin, x_axis, y_axis])


def _draw_plane_w1(ax: Axes3D.Axes) -> None:
    extent = _PLANE_EXTENT
    n = 15
    u = np.linspace(-extent, extent, n)
    v = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(u, v)
    W = np.ones_like(X)
    ax.plot_surface(X, Y, W, alpha=0.2, color="lightgray", edgecolor="none")


def _draw_3d_grid(ax: Axes3D.Axes, T: np.ndarray) -> None:
    vals = np.arange(-_GRID_RANGE, _GRID_RANGE + 1e-6, _GRID_STEP)
    w_vals = _GRID_W_VALS
    grey = "gray"
    alpha = 0.5
    lw = 1.0
    for y in vals:
        for w in w_vals:
            p0 = T @ np.array([-_GRID_RANGE, y, w])
            p1 = T @ np.array([_GRID_RANGE, y, w])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=grey, alpha=alpha, linewidth=lw)
    for x in vals:
        for w in w_vals:
            p0 = T @ np.array([x, -_GRID_RANGE, w])
            p1 = T @ np.array([x, _GRID_RANGE, w])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=grey, alpha=alpha, linewidth=lw)
    for x in vals:
        for y in vals:
            p0 = T @ np.array([x, y, 0.0])
            p1 = T @ np.array([x, y, 2.0])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=grey, alpha=alpha, linewidth=lw)


def _draw_3d_basis(ax: Axes3D.Axes, T: np.ndarray) -> None:
    o = np.array([0, 0, 0])
    t_x = T @ np.array([_AXIS_SCALE, 0, 0])
    t_y = T @ np.array([0, _AXIS_SCALE, 0])
    t_w = T @ np.array([0, 0, 1])
    ax.plot([o[0], t_x[0]], [o[1], t_x[1]], [o[2], t_x[2]], color="darkblue", linewidth=2.5)
    ax.plot([o[0], t_y[0]], [o[1], t_y[1]], [o[2], t_y[2]], color="darkgreen", linewidth=2.5)
    ax.plot([o[0], t_w[0]], [o[1], t_w[1]], [o[2], t_w[2]], color="darkred", linewidth=2.5)


def _draw_moving_grid(ax: Axes3D.Axes, T: np.ndarray) -> None:
    t_vals = np.linspace(-_GRID_RANGE, _GRID_RANGE, int(2 * _GRID_RANGE / _GRID_STEP) + 1)
    vals = np.arange(-_GRID_RANGE, _GRID_RANGE + 1e-6, _GRID_STEP)
    for x_fixed in vals:
        pts = T @ np.vstack([np.full_like(t_vals, x_fixed), t_vals, np.ones_like(t_vals)])
        ax.plot(pts[0], pts[1], pts[2], color="steelblue", alpha=0.6, linewidth=1)
    for y_fixed in vals:
        pts = T @ np.vstack([t_vals, np.full_like(t_vals, y_fixed), np.ones_like(t_vals)])
        ax.plot(pts[0], pts[1], pts[2], color="steelblue", alpha=0.6, linewidth=1)


def _draw_frame(ax: Axes3D.Axes, points: np.ndarray, color: str, label: str) -> None:
    o, px, py = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(*o, color=color, s=40, zorder=5)
    ax.plot([o[0], px[0]], [o[1], px[1]], [o[2], px[2]], color=color, linewidth=2.5, zorder=4)
    ax.plot([o[0], py[0]], [o[1], py[1]], [o[2], py[2]], color=color, linewidth=2.5, zorder=4)


def _draw_2d_grid(ax: plt.Axes, T: np.ndarray) -> None:
    vals = np.arange(-_GRID_RANGE, _GRID_RANGE + 1e-6, _GRID_STEP)
    t_vals = np.linspace(-_GRID_RANGE, _GRID_RANGE, int(2 * _GRID_RANGE / _GRID_STEP) + 1)
    for x_fixed in vals:
        pts = T @ np.vstack([np.full_like(t_vals, x_fixed), t_vals, np.ones_like(t_vals)])
        ax.plot(pts[0], pts[1], color="steelblue", alpha=0.6, linewidth=1)
    for y_fixed in vals:
        pts = T @ np.vstack([t_vals, np.full_like(t_vals, y_fixed), np.ones_like(t_vals)])
        ax.plot(pts[0], pts[1], color="steelblue", alpha=0.6, linewidth=1)


def _draw_2d_frame(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    o = points[0:2, 0]
    px = points[0:2, 1]
    py = points[0:2, 2]
    ax.scatter(*o, color=color, s=40, zorder=5)
    ax.plot([o[0], px[0]], [o[1], px[1]], color=color, linewidth=2.5, zorder=4)
    ax.plot([o[0], py[0]], [o[1], py[1]], color=color, linewidth=2.5, zorder=4)


def show_se2_homogeneous_interactive(
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    figsize: tuple[float, float] = (12, 5),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_2d = fig.add_subplot(122)
    ref_pts = _reference_frame_points()

    def update(x: float, y: float, theta: float):
        ax_3d.clear()
        ax_2d.clear()
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        T = _se2_matrix(x, y, theta)
        _draw_plane_w1(ax_3d)
        _draw_3d_grid(ax_3d, T)
        _draw_3d_basis(ax_3d, T)
        _draw_moving_grid(ax_3d, T)
        _draw_frame(ax_3d, ref_pts, color="gray", label="Reference")
        trans_pts = T @ ref_pts
        _draw_frame(ax_3d, trans_pts, color="steelblue", label="Transformed")
        ax_3d.set_xlabel("$x$")
        ax_3d.set_ylabel("$y$")
        ax_3d.set_zlabel("$w$")
        ax_3d.set_title("SE(2): homogeneous space")
        lim = _PLANE_EXTENT + 0.5
        ax_3d.set_xlim(-lim, lim)
        ax_3d.set_ylim(-lim, lim)
        ax_3d.set_zlim(-lim, lim)
        ax_3d.set_box_aspect((1, 1, 1))
        ax_3d.grid(False)
        _draw_2d_grid(ax_2d, T)
        _draw_2d_frame(ax_2d, ref_pts, color="gray")
        _draw_2d_frame(ax_2d, trans_pts, color="steelblue")
        ax_2d.set_xlabel("$x$")
        ax_2d.set_ylabel("$y$")
        ax_2d.set_title("Plane $w=1$ (xy)")
        ax_2d.set_aspect("equal")
        ax_2d.set_xlim(-lim, lim)
        ax_2d.set_ylim(-lim, lim)
        ax_2d.grid(True, alpha=0.3)
        fig.canvas.draw_idle()
        return fig

    sliders = {
        "x": widgets.FloatSlider(value=0, min=x_range[0], max=x_range[1], step=0.1, description="x"),
        "y": widgets.FloatSlider(value=0, min=y_range[0], max=y_range[1], step=0.1, description="y"),
        "theta": widgets.FloatSlider(value=0, min=0, max=2 * np.pi, step=0.05, description="θ"),
    }
    ui = widgets.HBox([sliders["x"], sliders["y"], sliders["theta"]])
    out = widgets.interactive_output(update, sliders)
    display(ui, out)
    update(0.0, 0.0, 0.0)
