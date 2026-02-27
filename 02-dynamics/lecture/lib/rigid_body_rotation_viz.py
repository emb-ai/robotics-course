from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ipywidgets import interact, FloatSlider, IntSlider

from .integrators import rk4
from .plot_utils import set_3d_equal_aspect
from .rigid_body_dynamics import (
    euler_rotation_equations,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
    quat_derivative,
)


def _full_rotation_rhs(I1: float, I2: float, I3: float):
    I_body = np.diag([I1, I2, I3])
    I_body_inv = np.diag([1 / I1, 1 / I2, 1 / I3])

    def rhs(t, state):
        q = quat_normalize(state[:4])
        L_world = state[4:7]
        R = quat_to_rotation_matrix(q)
        omega_world = R @ I_body_inv @ R.T @ L_world
        omega_body = R.T @ omega_world
        dq = quat_derivative(q, omega_body)
        dL = np.zeros(3)
        return np.concatenate([dq, dL])
    return rhs


def show_dzhanibekov_effect(
    I1: float = 1.0, I2: float = 2.0, I3: float = 3.0,
    perturbation: float = 0.1, T: float = 20.0, h: float = 0.005,
):
    # initial rotation near intermediate axis (axis 2)
    omega0_body = np.array([perturbation, 1.0, perturbation])
    I_body = np.diag([I1, I2, I3])
    L0_world = I_body @ omega0_body
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    state0 = np.concatenate([q0, L0_world])

    rhs = _full_rotation_rhs(I1, I2, I3)
    res = rk4(rhs, state0, (0.0, T), h)

    # renormalize quaternions
    for i in range(len(res.t)):
        res.y[i, :4] = quat_normalize(res.y[i, :4])

    # recover omega in body frame at each step
    I_body_inv = np.diag([1 / I1, 1 / I2, 1 / I3])
    omegas_body = np.empty((len(res.t), 3))
    for i in range(len(res.t)):
        R = quat_to_rotation_matrix(res.y[i, :4])
        L_w = res.y[i, 4:7]
        omega_w = R @ I_body_inv @ R.T @ L_w
        omegas_body[i] = R.T @ omega_w

    # energy and angular momentum magnitude
    kinetic = np.array([
        0.5 * (I1 * w[0]**2 + I2 * w[1]**2 + I3 * w[2]**2)
        for w in omegas_body
    ])
    L_mag = np.linalg.norm(res.y[:, 4:7], axis=1)

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # omega components vs time
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["ω₁ (min I)", "ω₂ (mid I)", "ω₃ (max I)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for j in range(3):
        ax1.plot(res.t, omegas_body[:, j], color=colors[j], lw=1.2, label=labels[j])
    ax1.set_xlabel("t")
    ax1.set_ylabel("ω (body frame)")
    ax1.set_title("Dzhanibekov effect: rotation near intermediate axis")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 3D trajectory of body-frame x-axis in world frame
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    body_axes = np.eye(3)
    all_pts = []
    for j, (label, color) in enumerate(zip(["body x", "body y", "body z"], colors)):
        world_dirs = np.array([
            quat_to_rotation_matrix(res.y[i, :4]) @ body_axes[j]
            for i in range(0, len(res.t), max(1, len(res.t) // 500))
        ])
        all_pts.append(world_dirs)
        ax2.plot(
            world_dirs[:, 0], world_dirs[:, 1], world_dirs[:, 2],
            color=color, lw=0.8, alpha=0.7, label=label,
        )
    all_pts = np.concatenate(all_pts, axis=0)
    set_3d_equal_aspect(ax2, all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], margin=0.1)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Body axes in world frame")
    ax2.legend(fontsize=7)

    # energy conservation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(res.t, kinetic, "k-", lw=1.2)
    ax3.set_xlabel("t")
    ax3.set_ylabel("T")
    ax3.set_title(f"Kinetic energy (relative drift: {abs(kinetic[-1]/kinetic[0]-1):.2e})")
    ax3.grid(True, alpha=0.3)

    # |L| conservation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(res.t, L_mag, "k-", lw=1.2)
    ax4.set_xlabel("t")
    ax4.set_ylabel("|L|")
    ax4.set_title(f"|L| conservation (relative drift: {abs(L_mag[-1]/L_mag[0]-1):.2e})")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_euler_equations_stability(
    I1: float = 1.0, I2: float = 2.0, I3: float = 3.0,
    T: float = 30.0, h: float = 0.005,
):
    rhs = euler_rotation_equations(I1, I2, I3)
    perturbation = 0.05

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [
        f"Near axis 1 (I={I1}, stable)",
        f"Near axis 2 (I={I2}, UNSTABLE)",
        f"Near axis 3 (I={I3}, stable)",
    ]
    initial_conditions = [
        np.array([1.0, perturbation, perturbation]),
        np.array([perturbation, 1.0, perturbation]),
        np.array([perturbation, perturbation, 1.0]),
    ]

    for ax, title, w0 in zip(axes, titles, initial_conditions):
        res = rk4(rhs, w0, (0.0, T), h)
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        for j in range(3):
            ax.plot(res.t, res.y[:, j], color=colors[j], lw=1.0, label=f"ω{j+1}")
        ax.set_xlabel("t")
        ax.set_ylabel("ω")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Tennis racket theorem: intermediate axis is unstable", fontsize=12, y=1.02)
    fig.tight_layout()
    plt.show()


def show_dzhanibekov_interactive():
    @interact(
        I2=FloatSlider(min=1.1, max=5.0, step=0.1, value=2.0, description="I₂"),
        perturbation=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description="pert"),
    )
    def _update(I2, perturbation):
        show_dzhanibekov_effect(I1=1.0, I2=I2, I3=5.0, perturbation=perturbation)
