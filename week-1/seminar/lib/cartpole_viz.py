from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_CYL_R = 0.5
_X_LIM = 2.0


def _draw_cartpole_workspace(ax: plt.Axes, x: float, theta: float, pole_len: float = 0.5) -> None:
    cart_w, cart_h = 0.3, 0.15
    ws_left = -_X_LIM - pole_len
    ws_right = _X_LIM + pole_len
    ws_bottom = -pole_len
    ws_top = pole_len
    ax.set_xlim(ws_left, ws_right)
    ax.set_ylim(ws_bottom * 1.5, ws_top * 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Cart–pole")
    ws_rect = Rectangle((ws_left, ws_bottom), ws_right - ws_left, ws_top - ws_bottom,
                        facecolor="lightgray", edgecolor="none", alpha=0.25, linewidth=1.5,
                        label="Workspace", zorder=0)
    ax.add_patch(ws_rect)
    rect = Rectangle((x - cart_w / 2, -cart_h / 2), cart_w, cart_h, facecolor="steelblue", edgecolor="black", zorder=2)
    ax.add_patch(rect)
    tip_x = x + pole_len * np.sin(theta)
    tip_z = pole_len * np.cos(theta)
    ax.plot([x, tip_x], [0, tip_z], "o-", color="black", linewidth=2, markersize=6)
    ax.plot(tip_x, tip_z, "r*", markersize=15, label="End-effector")


def _draw_config_space_cartpole(ax: plt.Axes) -> None:
    ax.set_xlim(-_X_LIM, _X_LIM)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("auto")
    ax.set_xlabel("$x$ (position)")
    ax.set_ylabel(r"$\theta$ (rad)")
    ax.set_title(r"Configuration space $\mathbb{R} \times S^1$ (cylinder)")
    ax.axhline(-np.pi, color="gray", linewidth=0.8)
    ax.axhline(np.pi, color="gray", linewidth=0.8)
    arrow_kw = dict(mutation_scale=14, linewidth=1.5, shrinkA=0, shrinkB=0)
    arrow_dx = 0.3
    identified_sides_text_margin = -0.35
    ax.annotate("", xy=(-_X_LIM + arrow_dx, -np.pi), xytext=(_X_LIM - arrow_dx, -np.pi),
                arrowprops=dict(arrowstyle="->", color="blue", **arrow_kw))
    ax.annotate("", xy=(-_X_LIM + arrow_dx, np.pi), xytext=(_X_LIM - arrow_dx, np.pi),
                arrowprops=dict(arrowstyle="->", color="blue", **arrow_kw))
    ax.text(0, -(np.pi + identified_sides_text_margin), "A", fontsize=12, color="blue", va="top", ha="center", fontweight="bold")
    ax.text(0, np.pi + identified_sides_text_margin, "A", fontsize=12, color="blue", va="bottom", ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3)


def _cylinder_point(x: float, theta: float, R: float | None = None) -> tuple[float, float, float]:
    R = R if R is not None else _CYL_R
    return x, R * np.cos(theta), R * np.sin(theta)


def _draw_cylinder_3d(ax: plt.Axes, n_x: int = 20, n_theta: int = 32) -> None:
    R = _CYL_R
    x_ax = np.linspace(-_X_LIM, _X_LIM, n_x)
    theta = np.linspace(-np.pi, np.pi, n_theta)
    X, Theta = np.meshgrid(x_ax, theta)
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    ax.plot_surface(X, Y, Z, color="wheat", alpha=0.5, edgecolor="none")
    x_seam = np.linspace(-_X_LIM, _X_LIM, 50)
    ax.plot(x_seam, np.full_like(x_seam, -R), np.zeros_like(x_seam), color="blue", linewidth=2.5, label=r"seam $\theta=\pm\pi$")
    lim = max(_X_LIM, _CYL_R)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(r"Configuration on cylinder $\mathbb{R} \times S^1$")
    ax.set_box_aspect((1, 1, 1))


def show_cartpole_interactive(
    x: np.ndarray,
    pole_len: float = 0.5,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    from matplotlib import backend_bases
    _orig = getattr(backend_bases.NavigationToolbar2, "toolitems", None)
    if _orig:
        backend_bases.NavigationToolbar2.toolitems = tuple(
            t for t in _orig if (t[0] or "").lower() not in ("pan", "move")
        )
    x = np.asarray(x, dtype=float).ravel()
    if x.size != 2:
        raise ValueError("x must have length 2 (position, angle)")
    x[0] = np.clip(x[0], -_X_LIM, _X_LIM)
    x[1] = (x[1] + np.pi) % (2 * np.pi) - np.pi

    fig = plt.figure(figsize=figsize)
    ax_workspace = fig.add_subplot(131)
    ax_config = fig.add_subplot(132)
    ax_cyl = fig.add_subplot(133, projection="3d")

    def refresh() -> None:
        pos, theta = x[0], x[1]
        pos = np.clip(pos, -_X_LIM, _X_LIM)
        x[0] = pos
        ax_workspace.clear()
        _draw_cartpole_workspace(ax_workspace, pos, theta, pole_len)
        ax_workspace.legend(loc="upper right")
        ax_config.clear()
        _draw_config_space_cartpole(ax_config)
        theta_plot = (theta + np.pi) % (2 * np.pi) - np.pi
        ax_config.scatter([pos], [theta_plot], s=120, c="red", zorder=5, edgecolors="darkred")
        ax_cyl.clear()
        _draw_cylinder_3d(ax_cyl)
        xp, yp, zp = _cylinder_point(pos, theta)
        ax_cyl.scatter([xp], [yp], [zp], s=80, c="red", edgecolors="darkred")
        fig.canvas.draw_idle()

    def on_click_workspace(event) -> None:
        if event.inaxes != ax_workspace or event.button != 1:
            return
        x_click, z_click = event.xdata, event.ydata
        if x_click is None or z_click is None:
            return
        x_target = np.clip(x_click, -_X_LIM - pole_len, _X_LIM + pole_len)
        z_target = np.clip(z_click, -pole_len, pole_len)
        cos_theta = np.clip(z_target / pole_len, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        cart_x_right = x_target - pole_len * sin_theta
        cart_x_left = x_target + pole_len * sin_theta
        if abs(cart_x_right - x[0]) <= abs(cart_x_left - x[0]):
            x[0] = np.clip(cart_x_right, -_X_LIM, _X_LIM)
            x[1] = theta
        else:
            x[0] = np.clip(cart_x_left, -_X_LIM, _X_LIM)
            x[1] = -theta
        refresh()

    def on_click_config(event) -> None:
        if event.inaxes != ax_config or event.button != 1:
            return
        pos_click, theta_click = event.xdata, event.ydata
        if pos_click is None or theta_click is None:
            return
        x[0] = np.clip(pos_click, -_X_LIM, _X_LIM)
        x[1] = (theta_click + np.pi) % (2 * np.pi) - np.pi
        refresh()

    refresh()
    fig.canvas.mpl_connect("button_press_event", on_click_workspace)
    fig.canvas.mpl_connect("button_press_event", on_click_config)
    plt.tight_layout()
    plt.show()
