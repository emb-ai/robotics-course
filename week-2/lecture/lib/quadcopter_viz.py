from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Arc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .integrators import rk4
from .plot_utils import set_3d_equal_aspect


@dataclass
class QuadcopterParams:
    m: float = 0.5
    g: float = 9.81
    L: float = 0.25
    k: float = 2.98e-6
    b: float = 1.14e-7
    Ixx: float = 4.856e-3
    Iyy: float = 4.856e-3
    Izz: float = 8.801e-3
    k_d: float = 0.25


def _rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)
    return np.array([
        [ct * cs, sp * st * cs - cp * ss, cp * st * cs + sp * ss],
        [ct * ss, sp * st * ss + cp * cs, cp * st * ss - sp * cs],
        [-st,     sp * ct,                cp * ct],
    ])


def quadcopter_rhs(
    t: float,
    state: np.ndarray,
    params: QuadcopterParams,
    gamma: np.ndarray,
) -> np.ndarray:
    phi, theta, psi = state[3:6]
    vx, vy, vz = state[6:9]
    p, q, r = state[9:12]

    T_total = params.k * np.sum(gamma)
    tau_phi = params.L * params.k * (gamma[0] - gamma[2])
    tau_theta = params.L * params.k * (gamma[1] - gamma[3])
    tau_psi = params.b * (gamma[0] - gamma[1] + gamma[2] - gamma[3])

    R = _rotation_matrix(phi, theta, psi)
    thrust_world = R @ np.array([0.0, 0.0, T_total])

    gravity = np.array([0.0, 0.0, -params.m * params.g])
    drag = -params.k_d * np.array([vx, vy, vz])
    acc = (gravity + thrust_world + drag) / params.m

    dp = (tau_phi + (params.Iyy - params.Izz) * q * r) / params.Ixx
    dq = (tau_theta + (params.Izz - params.Ixx) * p * r) / params.Iyy
    dr = (tau_psi + (params.Ixx - params.Iyy) * p * q) / params.Izz

    cp_v, tp = np.cos(phi), np.tan(theta)
    sp_v = np.sin(phi)
    ct = np.cos(theta)
    ct_safe = ct if abs(ct) > 1e-12 else np.sign(ct) * 1e-12
    dphi = p + sp_v * tp * q + cp_v * tp * r
    dtheta = cp_v * q - sp_v * r
    dpsi = (sp_v * q + cp_v * r) / ct_safe

    return np.array([vx, vy, vz, dphi, dtheta, dpsi, acc[0], acc[1], acc[2], dp, dq, dr])


def show_quadcopter_diagram():
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(13, 5))

    L = 1.0

    ax = ax_top
    rotor_pos = {1: (-L, 0), 2: (0, L), 3: (L, 0), 4: (0, -L)}

    for i, (rx, ry) in rotor_pos.items():
        ax.plot([0, rx], [0, ry], "k-", lw=2.5, zorder=2)
        color = "#3498db" if i in (1, 3) else "#e74c3c"
        circ = Circle((rx, ry), 0.25, fc=color, ec="k", lw=1.2, alpha=0.3, zorder=3)
        ax.add_patch(circ)
        ax.text(rx, ry, str(i), ha="center", va="center", fontsize=12,
                weight="bold", zorder=4)
        cw = (i in (1, 3))
        arc = Arc((rx, ry), 0.45, 0.45, angle=0,
                  theta1=30 if cw else 150, theta2=330 if cw else 210,
                  color=color, lw=1.5)
        ax.add_patch(arc)
        tip_angle = np.radians(330 if cw else 150)
        ax.annotate("", xy=(rx + 0.225 * np.cos(tip_angle), ry + 0.225 * np.sin(tip_angle)),
                    xytext=(rx + 0.225 * np.cos(tip_angle - 0.3 * (1 if cw else -1)),
                            ry + 0.225 * np.sin(tip_angle - 0.3 * (1 if cw else -1))),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.annotate("", xy=(-L * 0.55, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.15", color="#e74c3c", lw=2))
    ax.text(-L * 0.6, 0.1, "$x_B$", fontsize=13, color="#e74c3c")
    ax.annotate("", xy=(0, L * 0.55), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.15", color="#2ecc71", lw=2))
    ax.text(0.08, L * 0.6, "$y_B$", fontsize=13, color="#2ecc71")

    ax.annotate("", xy=(0.05, 0), xytext=(-L + 0.05, 0),
                arrowprops=dict(arrowstyle="<->", color="#9b59b6", lw=1.2, ls="--"))
    ax.text(-L / 2, -0.18, "$L$", fontsize=13, color="#9b59b6", ha="center")

    ax.plot(0, 0, "ko", markersize=6, zorder=5)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_title("Top-down view", fontsize=12)
    ax.axis("off")

    ax = ax_side
    body_y = 0.0
    arm_hw = L
    ax.plot([-arm_hw, arm_hw], [body_y, body_y], "k-", lw=3, zorder=2)
    ax.plot(0, body_y, "ko", markersize=8, zorder=5)

    for label, rx in [("1", -arm_hw), ("3", arm_hw)]:
        circ = Circle((rx, body_y), 0.12, fc="#3498db", ec="k", lw=1, alpha=0.3, zorder=3)
        ax.add_patch(circ)
        arr_len = 0.6
        ax.annotate("", xy=(rx, body_y + arr_len), xytext=(rx, body_y + 0.12),
                    arrowprops=dict(arrowstyle="->,head_width=0.12", color="#e67e22", lw=2.5))
        ax.text(rx + 0.08, body_y + arr_len * 0.6, f"$T_{label}$", fontsize=12, color="#e67e22")

    grav_len = 0.5
    ax.annotate("", xy=(0, body_y - grav_len), xytext=(0, body_y - 0.05),
                arrowprops=dict(arrowstyle="->,head_width=0.12", color="#2c3e50", lw=2.5))
    ax.text(0.1, body_y - grav_len * 0.6, "$mg$", fontsize=13, color="#2c3e50")

    ax.annotate("", xy=(arm_hw + 0.15, body_y), xytext=(0, body_y),
                arrowprops=dict(arrowstyle="->,head_width=0.12", color="#e74c3c", lw=1.8))
    ax.text(arm_hw * 0.5, body_y + 0.1, "$x_B$", fontsize=12, color="#e74c3c")
    ax.annotate("", xy=(0, body_y + 0.9), xytext=(0, body_y),
                arrowprops=dict(arrowstyle="->,head_width=0.12", color="#3498db", lw=1.8))
    ax.text(0.08, body_y + 0.8, "$z_B$", fontsize=12, color="#3498db")

    arc_r = 0.35
    arc = Arc((0, body_y), arc_r * 2, arc_r * 2, angle=0,
              theta1=60, theta2=120, color="#9b59b6", lw=2)
    ax.add_patch(arc)
    tip = np.radians(60)
    ax.annotate("", xy=(arc_r * np.cos(tip), body_y + arc_r * np.sin(tip)),
                xytext=(arc_r * np.cos(tip + 0.15), body_y + arc_r * np.sin(tip + 0.15)),
                arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.5))
    ax.text(-0.15, body_y + arc_r + 0.1, r"$\tau_\phi$", fontsize=13, color="#9b59b6")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.9, 1.2)
    ax.set_aspect("equal")
    ax.set_title("Side view (rotor 1 ↔ rotor 3)", fontsize=12)
    ax.axis("off")

    fig.suptitle("Quadcopter schematic: rotors 1,3 (CW, blue) and 2,4 (CCW, red)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    plt.show()


def _hover_gamma(params: QuadcopterParams) -> float:
    return params.m * params.g / (4 * params.k)


def show_quadcopter_open_loop(
    active_rotors: tuple[int, ...] = (1, 2),
    T: float = 2.5,
    h: float = 0.001,
    params: QuadcopterParams | None = None,
):
    if params is None:
        params = QuadcopterParams()

    gamma_hover = _hover_gamma(params)
    gamma = np.zeros(4)
    for r in active_rotors:
        gamma[r - 1] = gamma_hover

    state0 = np.zeros(12)
    state0[2] = 10.0  # start at 10 m altitude

    def rhs(t, state):
        return quadcopter_rhs(t, state, params, gamma)

    res = rk4(rhs, state0, (0.0, T), h)
    ts, ys = res.t, res.y

    # stop plotting once the drone hits the ground
    ground_mask = ys[:, 2] >= 0.0
    last_valid = np.argmin(ground_mask) if not ground_mask.all() else len(ts)
    if last_valid == 0:
        last_valid = len(ts)
    ts = ts[:last_valid]
    ys = ys[:last_valid]

    active_str = ", ".join(str(r) for r in active_rotors)
    off_rotors = [r for r in [1, 2, 3, 4] if r not in active_rotors]
    off_str = ", ".join(str(r) for r in off_rotors)

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    # -- 3D trajectory --
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.plot(ys[:, 0], ys[:, 1], ys[:, 2], "k-", lw=1.0, alpha=0.6)
    ax1.plot([ys[0, 0]], [ys[0, 1]], [ys[0, 2]], "go", markersize=8, label="start")
    ax1.plot([ys[-1, 0]], [ys[-1, 1]], [ys[-1, 2]], "rs", markersize=8, label="end")

    n_arrows = min(12, len(ts))
    indices = np.linspace(0, len(ts) - 1, n_arrows, dtype=int)
    axis_len = 0.8
    for idx in indices:
        p = ys[idx, 0:3]
        R = _rotation_matrix(ys[idx, 3], ys[idx, 4], ys[idx, 5])
        for j, c in enumerate(colors):
            d = R[:, j] * axis_len
            ax1.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                       color=c, arrow_length_ratio=0.15, lw=1.0, alpha=0.6)

    xs = np.concatenate([ys[:, 0], [ys[:, 0].min() - axis_len, ys[:, 0].max() + axis_len]])
    ys_ext = np.concatenate([ys[:, 1], [ys[:, 1].min() - axis_len, ys[:, 1].max() + axis_len]])
    zs_ext = np.concatenate([ys[:, 2], [ys[:, 2].min() - axis_len, ys[:, 2].max() + axis_len]])
    set_3d_equal_aspect(ax1, xs, ys_ext, zs_ext, margin=0.1)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")
    ax1.set_title("3D trajectory + body axes")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for c, l in zip(colors, ["body x", "body y", "body z"])]
    handles += [Line2D([0], [0], marker="o", color="g", ls="", label="start"),
                Line2D([0], [0], marker="s", color="r", ls="", label="end")]
    ax1.legend(handles=handles, fontsize=7, loc="upper left")

    # -- attitude --
    ax2 = fig.add_subplot(gs[0, 1])
    angle_labels = [r"$\phi$ (roll)", r"$\theta$ (pitch)", r"$\psi$ (yaw)"]
    for j in range(3):
        ax2.plot(ts, np.degrees(ys[:, 3 + j]), color=colors[j], lw=1.2, label=angle_labels[j])
    ax2.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("angle [deg]")
    ax2.set_title("Attitude: the drone tips over")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # -- position --
    ax3 = fig.add_subplot(gs[1, 0])
    pos_labels = ["x (horizontal)", "y (horizontal)", "z (altitude)"]
    for j in range(3):
        ax3.plot(ts, ys[:, j], color=colors[j], lw=1.2, label=pos_labels[j])
    ax3.axhline(0, color="k", ls=":", lw=0.8, alpha=0.3)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("position [m]")
    ax3.set_title("Position: falls and drifts")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # -- angular velocity --
    ax4 = fig.add_subplot(gs[1, 1])
    w_labels = [r"$\omega_x$ (roll)", r"$\omega_y$ (pitch)", r"$\omega_z$ (yaw)"]
    for j in range(3):
        ax4.plot(ts, np.degrees(ys[:, 9 + j]), color=colors[j], lw=1.0, label=w_labels[j])
    ax4.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("angular velocity [deg/s]")
    ax4.set_title(r"Angular velocity: $\omega \times (I\omega)$ cross-coupling")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"Open-loop: rotors {active_str} ON at hover speed, rotors {off_str} OFF",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    # plt.show()
    return fig
