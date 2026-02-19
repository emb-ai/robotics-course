from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .rotations import euler_zyx, rodrigues
from .plotting import plot_frame_3d
from .se3 import se3_exp

_LIM = 1.5


def _setup_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-_LIM, _LIM)
    ax.set_ylim(-_LIM, _LIM)
    ax.set_zlim(-_LIM, _LIM)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    


def show_so3_euler_interactive(
    figsize: tuple[float, float] = (7, 6),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    moving_frame_colors = ["#FF00FF", "#FFFF00", "#00FFFF"]

    def update(alpha: float, beta: float, gamma: float):
        ax.clear()
        R = euler_zyx(alpha, beta, gamma)
        plot_frame_3d(np.eye(3), scale=1.0, ax=ax, label="Ref ")
        plot_frame_3d(R, scale=0.9, ax=ax, label="", colors=moving_frame_colors)
        _setup_axes(ax)
        ax.set_title("SO(3): Euler ZYX")
        fig.canvas.draw_idle()

    sliders = {
        "alpha": widgets.FloatSlider(
            value=0, min=0, max=2 * np.pi, step=0.05, description="α"
        ),
        "beta": widgets.FloatSlider(
            value=0,
            min=-np.pi / 2 + 0.01,
            max=np.pi / 2 - 0.01,
            step=0.05,
            description="β",
        ),
        "gamma": widgets.FloatSlider(
            value=0, min=0, max=2 * np.pi, step=0.05, description="γ"
        ),
    }
    ui = widgets.HBox([sliders["alpha"], sliders["beta"], sliders["gamma"]])
    out = widgets.interactive_output(update, sliders)
    display(ui, out)
    update(0.0, 0.0, 0.0)


def show_gimbal_lock_interactive(
    figsize: tuple[float, float] = (7, 6),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    moving_frame_colors = ["#FF00FF", "#FFFF00", "#00FFFF"]

    def update(beta_choice: str, alpha: float, gamma: float):
        ax.clear()
        beta = np.pi / 2 if beta_choice == "β = π/2" else -np.pi / 2
        R = euler_zyx(alpha, beta, gamma)
        plot_frame_3d(np.eye(3), scale=1.0, ax=ax, label="Ref ")
        plot_frame_3d(R, scale=0.9, ax=ax, label="", colors=moving_frame_colors)
        _setup_axes(ax)
        total = alpha + gamma
        ax.set_title(f"Gimbal lock: β = ±π/2 → only α+γ matters (α+γ = {total:.2f})")
        fig.canvas.draw_idle()

    sliders = {
        "beta_choice": widgets.RadioButtons(
            options=["β = π/2", "β = −π/2"],
            value="β = π/2",
            description="β",
        ),
        "alpha": widgets.FloatSlider(
            value=0.5, min=0, max=2 * np.pi, step=0.05, description="α"
        ),
        "gamma": widgets.FloatSlider(
            value=0.3, min=0, max=2 * np.pi, step=0.05, description="γ"
        ),
    }
    ui = widgets.VBox([
        sliders["beta_choice"],
        widgets.HBox([sliders["alpha"], sliders["gamma"]]),
    ])
    out = widgets.interactive_output(update, sliders)
    display(ui, out)
    update("β = π/2", 0.5, 0.3)


def show_axis_angle_interactive(
    figsize: tuple[float, float] = (7, 6),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def update(omega_x: float, omega_y: float, omega_z: float, theta: float):
        ax.clear()
        axis = np.array([omega_x, omega_y, omega_z], dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / n
        R = rodrigues(axis, theta)
        plot_frame_3d(np.eye(3), scale=1.0, ax=ax, label="Ref ")
        plot_frame_3d(R, scale=0.9, ax=ax, label="")
        _setup_axes(ax)
        ax.set_title("SO(3): Axis–angle (Rodrigues)")
        fig.canvas.draw_idle()

    sliders = {
        "omega_x": widgets.FloatSlider(
            value=1, min=-1, max=1, step=0.1, description="ωx"
        ),
        "omega_y": widgets.FloatSlider(
            value=0, min=-1, max=1, step=0.1, description="ωy"
        ),
        "omega_z": widgets.FloatSlider(
            value=0, min=-1, max=1, step=0.1, description="ωz"
        ),
        "theta": widgets.FloatSlider(
            value=0.5, min=0, max=2 * np.pi, step=0.05, description="θ"
        ),
    }
    ui = widgets.HBox(
        [sliders["omega_x"], sliders["omega_y"], sliders["omega_z"], sliders["theta"]]
    )
    out = widgets.interactive_output(update, sliders)
    display(ui, out)
    update(1.0, 0.0, 0.0, 0.5)


def show_se3_interactive(
    figsize: tuple[float, float] = (7, 6),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def update(
        px: float, py: float, pz: float, alpha: float, beta: float, gamma: float
    ):
        ax.clear()
        R = euler_zyx(alpha, beta, gamma)
        p = np.array([px, py, pz])
        plot_frame_3d(np.eye(3), origin=np.zeros(3), scale=1.0, ax=ax, label="World ")
        plot_frame_3d(R, origin=p, scale=0.6, ax=ax, label="")
        _setup_axes(ax)
        ax.set_title("SE(3): translation + Euler rotation")
        fig.canvas.draw_idle()

    sliders = {
        "px": widgets.FloatSlider(
            value=0, min=-_LIM, max=_LIM, step=0.1, description="px"
        ),
        "py": widgets.FloatSlider(
            value=0, min=-_LIM, max=_LIM, step=0.1, description="py"
        ),
        "pz": widgets.FloatSlider(
            value=0, min=-_LIM, max=_LIM, step=0.1, description="pz"
        ),
        "alpha": widgets.FloatSlider(
            value=0, min=0, max=2 * np.pi, step=0.05, description="α"
        ),
        "beta": widgets.FloatSlider(
            value=0,
            min=-np.pi / 2 + 0.01,
            max=np.pi / 2 - 0.01,
            step=0.05,
            description="β",
        ),
        "gamma": widgets.FloatSlider(
            value=0, min=0, max=2 * np.pi, step=0.05, description="γ"
        ),
    }
    ui = widgets.VBox(
        [
            widgets.HBox([sliders["px"], sliders["py"], sliders["pz"]]),
            widgets.HBox([sliders["alpha"], sliders["beta"], sliders["gamma"]]),
        ]
    )
    out = widgets.interactive_output(update, sliders)
    display(ui, out)
    update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def show_screw_motion_interactive(
    n_frames: int = 6,
    figsize: tuple[float, float] = (7, 6),
) -> None:
    import ipywidgets as widgets
    from IPython.display import display

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def update(
        wx: float,
        wy: float,
        wz: float,
        vx: float,
        vy: float,
        vz: float,
        theta_max: float,
    ):
        ax.clear()
        S = np.array([wx, wy, wz, vx, vy, vz], dtype=float)
        thetas = np.linspace(0, max(theta_max, 0.01), n_frames)
        for i, t in enumerate(thetas):
            T = se3_exp(S, t)
            R = T[:3, :3]
            p = T[:3, 3]
            scale = 0.4 if n_frames > 1 else 0.8
            plot_frame_3d(R, origin=p, scale=scale, ax=ax, label="")
        _setup_axes(ax)
        ax.set_title("Screw motion: T(θ) = exp(S θ)")
        fig.canvas.draw_idle()

    sliders = {
        "wx": widgets.FloatSlider(value=0, min=-1, max=1, step=0.1, description="ωx"),
        "wy": widgets.FloatSlider(value=0, min=-1, max=1, step=0.1, description="ωy"),
        "wz": widgets.FloatSlider(value=1, min=-1, max=1, step=0.1, description="ωz"),
        "vx": widgets.FloatSlider(value=0, min=-1, max=1, step=0.1, description="vx"),
        "vy": widgets.FloatSlider(value=0.5, min=-1, max=1, step=0.1, description="vy"),
        "vz": widgets.FloatSlider(value=0, min=-1, max=1, step=0.1, description="vz"),
        "theta_max": widgets.FloatSlider(
            value=np.pi, min=0.1, max=2 * np.pi, step=0.1, description="θ max"
        ),
    }
    ui = widgets.VBox(
        [
            widgets.HBox([sliders["wx"], sliders["wy"], sliders["wz"]]),
            widgets.HBox([sliders["vx"], sliders["vy"], sliders["vz"]]),
            sliders["theta_max"],
        ]
    )
    out = widgets.interactive_output(update, sliders)
    display(ui, out)
    update(0.0, 0.0, 1.0, 0.0, 0.5, 0.0, np.pi)
