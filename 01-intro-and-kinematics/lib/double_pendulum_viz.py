from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Wedge

from .plotting import plot_planar_robot

_TORUS_R, _TORUS_r = 1.2, 0.4


def _ik_2r_analytic(x: float, y: float, L1: float, L2: float) -> list[tuple[float, float]]:
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(1 - c2 * c2)
    solutions = []
    for sign in (1, -1):
        theta2 = np.arctan2(sign * s2, c2)
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        solutions.append((theta1, theta2))
    return solutions


def _project_to_workspace(x: float, y: float, R_inner: float, R_outer: float) -> tuple[float, float]:
    r = np.hypot(x, y)
    if r <= 1e-10:
        return R_inner, 0.0
    if r < R_inner:
        s = R_inner / r
        return x * s, y * s
    if r > R_outer:
        s = R_outer / r
        return x * s, y * s
    return x, y


def _angular_diff(a: float, b: float) -> float:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _pick_closest_ik_solution(
    solutions: list[tuple[float, float]], current: np.ndarray
) -> np.ndarray:
    if not solutions:
        return current
    best = solutions[0]
    best_dist = sum(_angular_diff(solutions[0][i], current[i]) ** 2 for i in range(2))
    for sol in solutions[1:]:
        d = sum(_angular_diff(sol[i], current[i]) ** 2 for i in range(2))
        if d < best_dist:
            best_dist = d
            best = sol
    return np.array(best)


def _dist_theta1_to_bottom_semicircle(t1: float) -> float:
    t1 = (t1 + np.pi) % (2 * np.pi) - np.pi
    if -np.pi <= t1 <= 0:
        return 0.0
    return min(t1, np.pi - t1)


def _pick_constrained_ik_solution(
    solutions: list[tuple[float, float]], current: np.ndarray, t1_lo: float, t1_hi: float
) -> np.ndarray:
    if not solutions:
        return current
    valid = [s for s in solutions if t1_lo <= s[0] <= t1_hi]
    if valid:
        return _pick_closest_ik_solution(valid, current)
    best = min(
        solutions,
        key=lambda s: _dist_theta1_to_bottom_semicircle(s[0])
    )
    t1 = (best[0] + np.pi) % (2 * np.pi) - np.pi
    if t1_lo <= t1 <= t1_hi:
        t1_clamped = t1
    else:
        d_hi = abs(_angular_diff(t1, t1_hi))
        d_lo = abs(_angular_diff(t1, t1_lo))
        t1_clamped = t1_hi if d_hi <= d_lo else t1_lo
    return np.array([t1_clamped, best[1]])


def _draw_workspace(ax: plt.Axes, R_inner: float, R_outer: float) -> None:
    outer = Circle(
        (0, 0), R_outer, fill=True, facecolor="lightgray", edgecolor="none", alpha=0.4, zorder=0
    )
    ax.add_patch(outer)
    if R_inner > 1e-6:
        inner = Circle(
            (0, 0), R_inner, fill=True, facecolor="white", edgecolor="none", zorder=1
        )
        ax.add_patch(inner)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Double Pendulum")


def _draw_workspace_constrained(ax: plt.Axes, R_inner: float, R_outer: float, L1: float, L2: float) -> None:
    bottom_outer = Wedge(
        (0, 0), R_outer, 180, 360, fill=True, facecolor="lightgray", edgecolor="none", alpha=0.4, zorder=0
    )
    ax.add_patch(bottom_outer)
    if R_inner > 1e-6:
        bottom_inner = Wedge(
            (0, 0), R_inner, 180, 360, fill=True, facecolor="white", edgecolor="none", zorder=1
        )
        ax.add_patch(bottom_inner)
    top_left = Wedge(
        (-L1, 0), L2, 0, 180, fill=True, facecolor="lightgray", edgecolor="none", alpha=0.4, zorder=0
    )
    ax.add_patch(top_left)
    top_right = Wedge(
        (L1, 0), L2, 0, 180, fill=True, facecolor="lightgray", edgecolor="none", alpha=0.4, zorder=0
    )
    ax.add_patch(top_right)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Constrained Double Pendulum")


def _draw_config_space(ax: plt.Axes) -> None:
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$ (rad)")
    ax.set_ylabel(r"$\theta_2$ (rad)")
    ax.set_title(r"""Configuration space $\mathbb{T}^2$
    (sides glued → flat torus)""")
    ax.axhline(-np.pi, color="gray", linewidth=0.8)
    ax.axhline(np.pi, color="gray", linewidth=0.8)
    ax.axvline(-np.pi, color="gray", linewidth=0.8)
    ax.axvline(np.pi, color="gray", linewidth=0.8)
    arrow_kw = dict(mutation_scale=14, linewidth=1.5, shrinkA=0, shrinkB=0)
    identified_sides_text_margin = -0.35
    ax.annotate("", xy=(-np.pi, -np.pi + 0.5), xytext=(-np.pi, np.pi - 0.5),
                arrowprops=dict(arrowstyle="->", color="red", **arrow_kw))
    ax.text(-(np.pi + identified_sides_text_margin), 0, "B", fontsize=12, color="red", va="center", ha="right", fontweight="bold")
    ax.annotate("", xy=(np.pi, -np.pi + 0.5), xytext=(np.pi, np.pi - 0.5),
                arrowprops=dict(arrowstyle="->", color="red", **arrow_kw))
    ax.text(np.pi + identified_sides_text_margin, 0, "B", fontsize=12, color="red", va="center", ha="left", fontweight="bold")
    ax.annotate("", xy=(-np.pi + 0.5, -np.pi), xytext=(np.pi - 0.5, -np.pi),
                arrowprops=dict(arrowstyle="->", color="blue", **arrow_kw))
    ax.text(0, -(np.pi + identified_sides_text_margin), "A", fontsize=12, color="blue", ha="center", va="top", fontweight="bold")
    ax.annotate("", xy=(-np.pi + 0.5, np.pi), xytext=(np.pi - 0.5, np.pi),
                arrowprops=dict(arrowstyle="->", color="blue", **arrow_kw))
    ax.text(0, np.pi + identified_sides_text_margin, "A", fontsize=12, color="blue", ha="center", va="bottom", fontweight="bold")
    ax.grid(True, alpha=0.3)


def _torus_point(t1: float, t2: float, R: float | None = None, r: float | None = None) -> tuple[float, float, float]:
    R = R if R is not None else _TORUS_R
    r = r if r is not None else _TORUS_r
    x = (R + r * np.cos(t2)) * np.cos(t1)
    y = (R + r * np.cos(t2)) * np.sin(t1)
    z = r * np.sin(t2)
    return x, y, z


def _draw_torus_3d(ax: plt.Axes, n_u: int = 32, n_v: int = 24) -> None:
    R, r = _TORUS_R, _TORUS_r
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, color="wheat", alpha=0.5, edgecolor="none")
    phi = np.linspace(0, 2 * np.pi, 80)
    ax.plot(-(R + r * np.cos(phi)), np.zeros_like(phi), r * np.sin(phi), color="red", linewidth=2, label="B ($\\theta_1=\\pi$)")
    ax.plot((R - r) * np.cos(phi), (R - r) * np.sin(phi), np.zeros_like(phi), color="blue", linewidth=2, label="A ($\\theta_2=\\pi$)")
    lim = R + r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title("Configuration on torus $\\mathbb{T}^2$")
    ax.set_box_aspect((1, 1, 1))


def show_double_pendulum_interactive(
    L1: float, L2: float, theta: np.ndarray, figsize: tuple[float, float] = (12, 4)
) -> plt.Figure:
    from matplotlib import backend_bases
    _orig = getattr(backend_bases.NavigationToolbar2, "toolitems", None)
    if _orig:
        backend_bases.NavigationToolbar2.toolitems = tuple(
            t for t in _orig if (t[0] or "").lower() not in ("pan", "move")
        )
    R_inner = abs(L1 - L2)
    R_outer = L1 + L2
    theta = np.asarray(theta, dtype=float)
    if theta.size != 2:
        raise ValueError("theta must have length 2")
    theta = theta.ravel()

    fig = plt.figure(figsize=figsize)
    ax_pendulum = fig.add_subplot(131)
    ax_config = fig.add_subplot(132)
    ax_torus = fig.add_subplot(133, projection="3d")

    def refresh() -> None:
        t1, t2 = theta[0], theta[1]
        ax_pendulum.clear()
        _draw_workspace(ax_pendulum, R_inner, R_outer)
        plot_planar_robot(theta, L=[L1, L2], ax=ax_pendulum)
        handles, labels = ax_pendulum.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor="lightgray", edgecolor="gray", alpha=0.4))
        labels.insert(0, "End-effector workspace")
        ax_pendulum.legend(handles, labels, loc="upper right")
        ax_config.clear()
        _draw_config_space(ax_config)
        t1_plot = (t1 + np.pi) % (2 * np.pi) - np.pi
        t2_plot = (t2 + np.pi) % (2 * np.pi) - np.pi
        ax_config.scatter([t1_plot], [t2_plot], s=120, c="red", zorder=5, edgecolors="darkred")
        ax_torus.clear()
        _draw_torus_3d(ax_torus)
        xp, yp, zp = _torus_point(t1, t2)
        ax_torus.scatter([xp], [yp], [zp], s=80, c="red", edgecolors="darkred")
        fig.canvas.draw_idle()

    def on_click_workspace(event) -> None:
        if event.inaxes != ax_pendulum or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        xp, yp = _project_to_workspace(x, y, R_inner, R_outer)
        sols = _ik_2r_analytic(xp, yp, L1, L2)
        if not sols:
            return
        theta[:] = _pick_closest_ik_solution(sols, theta)
        refresh()

    def on_click_config(event) -> None:
        if event.inaxes != ax_config or event.button != 1:
            return
        t1, t2 = event.xdata, event.ydata
        if t1 is None or t2 is None:
            return
        theta[0] = (t1 + np.pi) % (2 * np.pi) - np.pi
        theta[1] = (t2 + np.pi) % (2 * np.pi) - np.pi
        refresh()

    refresh()
    fig.canvas.mpl_connect("button_press_event", on_click_workspace)
    fig.canvas.mpl_connect("button_press_event", on_click_config)
    plt.tight_layout()
    plt.show()


_THETA1_CONSTRAINED_LO = -np.pi
_THETA1_CONSTRAINED_HI = 0.0


def _draw_config_space_constrained(ax: plt.Axes) -> None:
    ax.set_xlim(_THETA1_CONSTRAINED_LO, _THETA1_CONSTRAINED_HI)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$ (rad)")
    ax.set_ylabel(r"$\theta_2$ (rad)")
    ax.set_title(r"Configuration space (half torus): $\theta_1 \in [-\pi,0]$")
    ax.axhline(-np.pi, color="gray", linewidth=0.8)
    ax.axhline(np.pi, color="gray", linewidth=0.8)
    ax.axvline(_THETA1_CONSTRAINED_LO, color="gray", linewidth=0.8)
    ax.axvline(_THETA1_CONSTRAINED_HI, color="gray", linewidth=0.8)
    arrow_kw = dict(mutation_scale=14, linewidth=1.5, shrinkA=0, shrinkB=0)
    identified_sides_text_margin = -0.35
    ax.annotate("", xy=(_THETA1_CONSTRAINED_LO + 0.5, -np.pi), xytext=(_THETA1_CONSTRAINED_HI - 0.5, -np.pi),
                arrowprops=dict(arrowstyle="->", color="blue", **arrow_kw))
    ax.text(-np.pi / 2, -(np.pi + identified_sides_text_margin), "A", fontsize=12, color="blue", ha="center", va="top", fontweight="bold")
    ax.annotate("", xy=(_THETA1_CONSTRAINED_LO + 0.5, np.pi), xytext=(_THETA1_CONSTRAINED_HI - 0.5, np.pi),
                arrowprops=dict(arrowstyle="->", color="blue", **arrow_kw))
    ax.text(-np.pi / 2, np.pi + identified_sides_text_margin, "A", fontsize=12, color="blue", ha="center", va="bottom", fontweight="bold")
    ax.grid(True, alpha=0.3)


def _half_torus_point(t1: float, t2: float, R: float | None = None, r: float | None = None) -> tuple[float, float, float]:
    R = R if R is not None else _TORUS_R
    r = r if r is not None else _TORUS_r
    x = (R + r * np.cos(t2)) * np.cos(t1)
    y = (R + r * np.cos(t2)) * np.sin(t1)
    z = r * np.sin(t2)
    return x, y, z


def _draw_half_torus_3d(ax: plt.Axes, n_u: int = 20, n_v: int = 32) -> None:
    R, r = _TORUS_R, _TORUS_r
    u = np.linspace(_THETA1_CONSTRAINED_LO, _THETA1_CONSTRAINED_HI, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, color="wheat", alpha=0.5, edgecolor="none")
    phi = np.linspace(np.pi, 2 * np.pi, 80)
    ax.plot((R - r) * np.cos(phi), (R - r) * np.sin(phi), np.zeros_like(phi), color="blue", linewidth=2, label="A ($\\theta_1=\\pi$)")
    lim = R + r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(r"Configuration on half torus ($\theta_1 \in [-\pi,0]$)")
    ax.set_box_aspect((1, 1, 1))


def show_constrained_double_pendulum_interactive(
    L1: float, L2: float, theta: np.ndarray, figsize: tuple[float, float] = (12, 4)
) -> plt.Figure:
    from matplotlib import backend_bases
    _orig = getattr(backend_bases.NavigationToolbar2, "toolitems", None)
    if _orig:
        backend_bases.NavigationToolbar2.toolitems = tuple(
            t for t in _orig if (t[0] or "").lower() not in ("pan", "move")
        )
    R_inner = abs(L1 - L2)
    R_outer = L1 + L2
    theta = np.asarray(theta, dtype=float)
    if theta.size != 2:
        raise ValueError("theta must have length 2")
    theta = theta.ravel()
    theta[0] = np.clip(theta[0], _THETA1_CONSTRAINED_LO, _THETA1_CONSTRAINED_HI)

    fig = plt.figure(figsize=figsize)
    ax_pendulum = fig.add_subplot(131)
    ax_config = fig.add_subplot(132)
    ax_torus = fig.add_subplot(133, projection="3d")

    def refresh() -> None:
        theta[0] = np.clip(theta[0], _THETA1_CONSTRAINED_LO, _THETA1_CONSTRAINED_HI)
        t1, t2 = theta[0], theta[1]
        ax_pendulum.clear()
        _draw_workspace_constrained(ax_pendulum, R_inner, R_outer, L1, L2)
        plot_planar_robot(theta, L=[L1, L2], ax=ax_pendulum)
        handles, labels = ax_pendulum.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor="lightgray", edgecolor="gray", alpha=0.4))
        labels.insert(0, "End-effector workspace")
        ax_pendulum.legend(handles, labels, loc="upper right")
        ax_config.clear()
        _draw_config_space_constrained(ax_config)
        t2_plot = (t2 + np.pi) % (2 * np.pi) - np.pi
        ax_config.scatter([t1], [t2_plot], s=120, c="red", zorder=5, edgecolors="darkred")
        ax_torus.clear()
        _draw_half_torus_3d(ax_torus)
        xp, yp, zp = _half_torus_point(t1, t2)
        ax_torus.scatter([xp], [yp], [zp], s=80, c="red", edgecolors="darkred")
        fig.canvas.draw_idle()

    def on_click_workspace(event) -> None:
        if event.inaxes != ax_pendulum or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        xp, yp = _project_to_workspace(x, y, R_inner, R_outer)
        sols = _ik_2r_analytic(xp, yp, L1, L2)
        if not sols:
            return
        theta[:] = _pick_constrained_ik_solution(
            sols, theta, _THETA1_CONSTRAINED_LO, _THETA1_CONSTRAINED_HI
        )
        refresh()

    def on_click_config(event) -> None:
        if event.inaxes != ax_config or event.button != 1:
            return
        t1, t2 = event.xdata, event.ydata
        if t1 is None or t2 is None:
            return
        theta[0] = np.clip(t1, _THETA1_CONSTRAINED_LO, _THETA1_CONSTRAINED_HI)
        theta[1] = (t2 + np.pi) % (2 * np.pi) - np.pi
        refresh()

    refresh()
    fig.canvas.mpl_connect("button_press_event", on_click_workspace)
    fig.canvas.mpl_connect("button_press_event", on_click_config)
    plt.tight_layout()
    plt.show()
