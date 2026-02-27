from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

_TUBE_RADIUS = 1.0
_OUTER_RADIUS_MIN = 1.0
_OUTER_RADIUS_MAX = 2.5
_OUTER_RADIUS_INIT = _OUTER_RADIUS_MAX
_SPHERE_RADIUS = 1.0
_U_RES, _V_RES = 40, 30
_VIEW_LIM = 2.5


def _torus_mesh(
    major_radius: float,
    minor_radius: float,
    u_res: int = _U_RES,
    v_res: int = _V_RES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, u_res)
    v = np.linspace(0, 2 * np.pi, v_res)
    u, v = np.meshgrid(u, v)
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    return x, y, z


def _sphere_mesh(
    radius: float,
    u_res: int = _U_RES,
    v_res: int = _V_RES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, u_res)
    v = np.linspace(0, np.pi, v_res)
    u, v = np.meshgrid(u, v)
    x = radius * np.sin(v) * np.cos(u)
    y = radius * np.sin(v) * np.sin(u)
    z = radius * np.cos(v)
    return x, y, z


def _major_radius_from_outer_radius(outer_radius: float) -> float:
    return max(outer_radius - _TUBE_RADIUS, 0.0)


def _torus_curve_at_v(
    major_radius: float,
    minor_radius: float,
    v: float,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n, endpoint=True)
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = np.full_like(u, minor_radius * np.sin(v))
    return x, y, z


def _sphere_curve_at_v(radius: float, v: float, n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n, endpoint=True)
    x = radius * np.sin(v) * np.cos(u)
    y = radius * np.sin(v) * np.sin(u)
    z = np.full_like(u, radius * np.cos(v))
    return x, y, z


def _torus_point(major_radius: float, minor_radius: float, theta: float, phi: float) -> tuple[float, float, float]:
    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)
    return x, y, z


def _sphere_point(radius: float, theta: float, phi: float) -> tuple[float, float, float]:
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z


_CURVE_V = 0 + np.pi / 100
_POINT_THETA = np.pi * 1.9
_POINT_PHI = np.pi / 4


def coordinate_singularity(
    figsize: tuple[float, float] = (12, 5),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax_torus = fig.add_subplot(121, projection="3d", computed_zorder=False)
    ax_sphere = fig.add_subplot(122, projection="3d", computed_zorder=False)

    def draw_torus_extra(ax: plt.Axes, major: float) -> None:
        cx, cy, cz = _torus_curve_at_v(major, _TUBE_RADIUS, _CURVE_V + np.pi / 2)
        ax.plot(cx, cy, cz, color="darkred", linewidth=2, zorder=10)
        p1 = _torus_point(major, _TUBE_RADIUS, _POINT_THETA, _POINT_PHI)
        p2 = _torus_point(major, _TUBE_RADIUS, np.pi + _POINT_THETA, np.pi - _POINT_PHI)
        ax.scatter(*p1, color="black", s=80, zorder=100)
        ax.scatter(*p2, color="black", s=80, zorder=100)

    def draw_sphere_extra(ax: plt.Axes) -> None:
        cx, cy, cz = _sphere_curve_at_v(_SPHERE_RADIUS, _CURVE_V)
        ax.plot(cx, cy, cz, color="darkred", linewidth=2, zorder=10)
        p = _sphere_point(_SPHERE_RADIUS, _POINT_THETA, _POINT_PHI)
        ax.scatter(*p, color="black", s=80, zorder=100)

    major_init = _major_radius_from_outer_radius(_OUTER_RADIUS_INIT)
    tx, ty, tz = _torus_mesh(major_init, _TUBE_RADIUS)
    ax_torus.plot_surface(tx, ty, tz, color="steelblue", alpha=0.85, shade=True, zorder=1)
    draw_torus_extra(ax_torus, major_init)
    ax_torus.set_xlim(-_VIEW_LIM, _VIEW_LIM)
    ax_torus.set_ylim(-_VIEW_LIM, _VIEW_LIM)
    ax_torus.set_zlim(-_VIEW_LIM, _VIEW_LIM)
    ax_torus.set_xlabel("x")
    ax_torus.set_ylabel("y")
    ax_torus.set_zlabel("z")
    ax_torus.set_box_aspect((1, 1, 1))

    sx, sy, sz = _sphere_mesh(_SPHERE_RADIUS)
    ax_sphere.plot_surface(sx, sy, sz, color="coral", alpha=0.9, shade=True, zorder=1)
    draw_sphere_extra(ax_sphere)
    ax_sphere.set_xlim(-_VIEW_LIM, _VIEW_LIM)
    ax_sphere.set_ylim(-_VIEW_LIM, _VIEW_LIM)
    ax_sphere.set_zlim(-_VIEW_LIM, _VIEW_LIM)
    ax_sphere.set_xlabel("x")
    ax_sphere.set_ylabel("y")
    ax_sphere.set_zlabel("z")
    ax_sphere.set_box_aspect((1, 1, 1))

    def update(outer_radius: float) -> None:
        major = _major_radius_from_outer_radius(outer_radius)
        ax_torus.clear()
        tx, ty, tz = _torus_mesh(major, _TUBE_RADIUS)
        ax_torus.plot_surface(tx, ty, tz, color="steelblue", alpha=0.85, shade=True, zorder=1)
        draw_torus_extra(ax_torus, major)
        ax_torus.set_xlim(-_VIEW_LIM, _VIEW_LIM)
        ax_torus.set_ylim(-_VIEW_LIM, _VIEW_LIM)
        ax_torus.set_zlim(-_VIEW_LIM, _VIEW_LIM)
        ax_torus.set_xlabel("x")
        ax_torus.set_ylabel("y")
        ax_torus.set_zlabel("z")
        ax_torus.set_box_aspect((1, 1, 1))

        ax_sphere.clear()
        sx, sy, sz = _sphere_mesh(_SPHERE_RADIUS)
        ax_sphere.plot_surface(sx, sy, sz, color="coral", alpha=0.9, shade=True, zorder=1)
        draw_sphere_extra(ax_sphere)
        ax_sphere.set_xlim(-_VIEW_LIM, _VIEW_LIM)
        ax_sphere.set_ylim(-_VIEW_LIM, _VIEW_LIM)
        ax_sphere.set_zlim(-_VIEW_LIM, _VIEW_LIM)
        ax_sphere.set_xlabel("x")
        ax_sphere.set_ylabel("y")
        ax_sphere.set_zlabel("z")
        ax_sphere.set_box_aspect((1, 1, 1))
        fig.canvas.draw_idle()

    slider = widgets.FloatSlider(
        value=_OUTER_RADIUS_INIT,
        min=_OUTER_RADIUS_MIN,
        max=_OUTER_RADIUS_MAX,
        step=0.05,
        description="Outer radius",
    )
    out = widgets.interactive_output(update, {"outer_radius": slider})
    display(slider, out)
    update(_OUTER_RADIUS_INIT)
