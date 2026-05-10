"""Matplotlib-based visualiser for the 2D planar rocket with TVC."""
from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation

from .env import RocketParams

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotate(pts: np.ndarray, angle: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Rotate Nx2 points by *angle* (rad) around (cx, cy)."""
    c, s = np.cos(angle), np.sin(angle)
    pts = np.asarray(pts, dtype=float) - [cx, cy]
    rotated = pts @ np.array([[c, s], [-s, c]])
    return rotated + [cx, cy]


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_rocket(
    t: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    params: RocketParams,
    wind_fn: Callable[[float], float] | None = None,
    ref_states: np.ndarray | None = None,
    speed: float = 1.0,
) -> FuncAnimation:
    """Return a ``FuncAnimation`` of the rocket trajectory.

    Parameters match the output of ``simulate`` (controls may have one fewer
    row than states; the last control is repeated for display).
    """
    L = params.body_length
    W = params.body_width

    # --- subsample to ~200 frames ----------------------------------------
    n_total = len(t)
    target_frames = 200
    stride = max(1, n_total // target_frames)
    frame_idx = np.arange(0, n_total, stride)
    if frame_idx[-1] != n_total - 1:
        frame_idx = np.append(frame_idx, n_total - 1)
    n_frames = len(frame_idx)

    dt_sim = float(t[1] - t[0]) if len(t) > 1 else 0.01
    interval_ms = max(1, int(dt_sim * stride / speed * 1000))

    # --- pad controls to match states length -----------------------------
    if controls.shape[0] < states.shape[0]:
        pad = np.tile(controls[-1:], (states.shape[0] - controls.shape[0], 1))
        controls_ext = np.concatenate([controls, pad], axis=0)
    else:
        controls_ext = controls

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("z  [m]")

    # --- static scene elements -------------------------------------------
    xs, zs = states[:, 0], states[:, 1]
    x_min, x_max = xs.min(), xs.max()
    z_min, z_max = zs.min(), zs.max()
    pad_x = max(5.0, (x_max - x_min) * 0.25)
    pad_z = max(5.0, (z_max - z_min) * 0.15)
    view_x = (x_min - pad_x, x_max + pad_x)
    view_z = (min(z_min, 0) - pad_z * 0.5, z_max + pad_z)

    ground = mpatches.Rectangle(
        (view_x[0] - 10, view_z[0] - 10),
        (view_x[1] - view_x[0]) + 20,
        -view_z[0] + 10 + 10,
        facecolor="#2d5016", zorder=0,
    )
    ax.add_patch(ground)

    pad_w = 4.0
    landing_pad = mpatches.Rectangle(
        (-pad_w / 2, -0.15), pad_w, 0.15,
        facecolor="#888888", edgecolor="#aaaaaa", zorder=1,
    )
    ax.add_patch(landing_pad)

    ax.set_xlim(*view_x)
    ax.set_ylim(*view_z)

    # trajectory trace
    trail_line, = ax.plot([], [], lw=1.0, color="#00e5ff", alpha=0.6, zorder=2)

    # reference trajectory
    if ref_states is not None:
        ax.plot(
            ref_states[:, 0], ref_states[:, 1],
            ls="--", lw=0.8, color="#ffffff", alpha=0.25, zorder=1,
        )

    # time text
    time_txt = ax.text(
        0.97, 0.97, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color="white",
        fontfamily="monospace",
    )

    # wind arrow
    wind_arrow = None
    wind_txt = None
    if wind_fn is not None:
        wind_arrow = mpatches.FancyArrowPatch(
            (0, 0), (0, 0),
            arrowstyle="->,head_width=6,head_length=4",
            color="#ffeb3b", lw=2, zorder=5,
        )
        ax.add_patch(wind_arrow)
        wind_txt = ax.text(0, 0, "", fontsize=8, color="#ffeb3b", ha="center", va="bottom")

    # --- rocket artists --------------------------------------------------
    # Patches defined in body frame (CoG at origin). Each frame we apply an
    # Affine2D transform: rotate(-theta) then translate(x_w, z_w).
    # Positive theta = nose tilts toward +x world; Affine2D.rotate(+theta) would
    # tilt the wrong way for row-vector convention used in _rotate().

    fuselage = mpatches.FancyBboxPatch(
        (-W / 2, -L * 0.35), W, L * 0.7,
        boxstyle=mpatches.BoxStyle.Round(pad=0, rounding_size=W * 0.15),
        facecolor="#F0F0F0", edgecolor="#cccccc", lw=0.5, zorder=10,
    )
    ax.add_patch(fuselage)

    nose = mpatches.Ellipse(
        (0, L * 0.35), W, L * 0.36,
        facecolor="#E03030", edgecolor="#c02020", lw=0.5, zorder=11,
    )
    ax.add_patch(nose)

    porthole = mpatches.Circle(
        (0, L * 0.08), W * 0.18,
        facecolor="#4682B4", edgecolor="#333333", lw=1.0, zorder=12,
    )
    ax.add_patch(porthole)

    stripe = mpatches.Rectangle(
        (-W / 2, -L * 0.06), W, L * 0.02,
        facecolor="#2196F3", edgecolor="none", zorder=12,
    )
    ax.add_patch(stripe)

    fin_left = plt.Polygon(
        [[-W / 2, -L * 0.15], [-W * 1.1, -L * 0.38], [-W / 2, -L * 0.35]],
        closed=True, facecolor="#888888", edgecolor="#666666", lw=0.5, zorder=9,
    )
    ax.add_patch(fin_left)

    fin_right = plt.Polygon(
        [[W / 2, -L * 0.15], [W * 1.1, -L * 0.38], [W / 2, -L * 0.35]],
        closed=True, facecolor="#888888", edgecolor="#666666", lw=0.5, zorder=9,
    )
    ax.add_patch(fin_right)

    leg_left = mlines.Line2D([], [], color="#666666", lw=2, solid_capstyle="round", zorder=9)
    ax.add_line(leg_left)
    leg_right = mlines.Line2D([], [], color="#666666", lw=2, solid_capstyle="round", zorder=9)
    ax.add_line(leg_right)

    pivot_dot = mpatches.Circle((0, -L * 0.35), W * 0.08, facecolor="#444444", zorder=13)
    ax.add_patch(pivot_dot)

    nozzle = plt.Polygon(
        [[0, 0]] * 4, closed=True,
        facecolor="#505050", edgecolor="#3a3a3a", lw=0.5, zorder=8,
    )
    ax.add_patch(nozzle)

    flame = plt.Polygon(
        [[0, 0]] * 3, closed=True,
        facecolor="#FF8C00", edgecolor="#FF6600", lw=0.3, alpha=0.9, zorder=7,
    )
    ax.add_patch(flame)

    # Collect parametric patches that share the body-frame transform
    _body_patches = [fuselage, nose, porthole, stripe]

    # --- update function --------------------------------------------------
    trail_len = 200

    def _body_transform(theta: float, x_w: float, z_w: float) -> mtransforms.Affine2D:
        return (
            mtransforms.Affine2D()
            .rotate(-theta)
            .translate(x_w, z_w)
            + ax.transData
        )

    def _update(frame_num: int):
        idx = frame_idx[frame_num]
        x_w, z_w = states[idx, 0], states[idx, 1]
        theta = states[idx, 4]
        T_val, delta = controls_ext[idx]

        # trail
        lo = max(0, idx - trail_len * stride)
        trail_line.set_data(states[lo:idx + 1, 0], states[lo:idx + 1, 1])

        time_txt.set_text(f"t = {t[idx]:.2f} s")

        # wind
        if wind_fn is not None:
            w = wind_fn(t[idx])
            arrow_y = view_z[1] - pad_z * 0.3
            arrow_x0 = view_x[0] + pad_x * 0.3
            scale = 0.5
            wind_arrow.set_positions(
                (arrow_x0, arrow_y), (arrow_x0 + w * scale, arrow_y),
            )
            wind_txt.set_position((arrow_x0 + w * scale * 0.5, arrow_y + 0.5))
            wind_txt.set_text(f"wind {w:+.1f} m/s²")

        # --- body-frame transform for parametric patches ------------------
        tr = _body_transform(theta, x_w, z_w)
        for p in _body_patches:
            p.set_transform(tr)

        # --- vertex-based patches (Polygons, Lines) -----------------------
        fin_l_body = np.array([
            [-W / 2, -L * 0.15], [-W * 1.1, -L * 0.38], [-W / 2, -L * 0.35],
        ])
        fin_left.set_xy(_rotate(fin_l_body, -theta) + [x_w, z_w])

        fin_r_body = np.array([
            [W / 2, -L * 0.15], [W * 1.1, -L * 0.38], [W / 2, -L * 0.35],
        ])
        fin_right.set_xy(_rotate(fin_r_body, -theta) + [x_w, z_w])

        ll_s = _rotate(np.array([[-W / 2, -L * 0.35]]), -theta)[0] + [x_w, z_w]
        ll_e = _rotate(np.array([[-W * 0.9, -L * 0.47]]), -theta)[0] + [x_w, z_w]
        leg_left.set_data([ll_s[0], ll_e[0]], [ll_s[1], ll_e[1]])

        lr_s = _rotate(np.array([[W / 2, -L * 0.35]]), -theta)[0] + [x_w, z_w]
        lr_e = _rotate(np.array([[W * 0.9, -L * 0.47]]), -theta)[0] + [x_w, z_w]
        leg_right.set_data([lr_s[0], lr_e[0]], [lr_s[1], lr_e[1]])

        pivot_world = _rotate(np.array([[0, -L * 0.35]]), -theta)[0] + [x_w, z_w]
        pivot_dot.set_center(pivot_world)

        # --- nozzle: rotate by delta in local frame, then body transform --
        noz_local = np.array([
            [-W * 0.175, 0],
            [ W * 0.175, 0],
            [ W * 0.275, -L * 0.10],
            [-W * 0.275, -L * 0.10],
        ])
        noz_body = _rotate(noz_local, delta) + [0, -L * 0.35]
        nozzle.set_xy(_rotate(noz_body, -theta) + [x_w, z_w])

        # --- flame --------------------------------------------------------
        T_ratio = np.clip(T_val / params.T_max, 0, 1)
        flame_len = L * 0.2 + L * 1.0 * T_ratio
        flame_len *= 1.0 + 0.05 * (2 * np.random.random() - 1)  # flicker
        half_base = W * 0.2

        flame_local = np.array([
            [-half_base, -L * 0.10],
            [ half_base, -L * 0.10],
            [0,          -L * 0.10 - flame_len],
        ])
        flame_body = _rotate(flame_local, delta) + [0, -L * 0.35]
        flame.set_xy(_rotate(flame_body, -theta) + [x_w, z_w])
        flame.set_alpha(0.9 * T_ratio)

        return (
            trail_line, time_txt, fuselage, nose, porthole, stripe,
            fin_left, fin_right, leg_left, leg_right, pivot_dot,
            nozzle, flame,
        )

    anim = FuncAnimation(
        fig, _update, frames=n_frames, interval=interval_ms, blit=False,
    )
    return anim


# ---------------------------------------------------------------------------
# State / control plots
# ---------------------------------------------------------------------------

def plot_states(
    t: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    params: RocketParams,
    ref_states: np.ndarray | None = None,
    ref_controls: np.ndarray | None = None,
) -> plt.Figure:
    """Return a figure with 4x2 subplots of state and control histories."""

    t_ctrl = t[: controls.shape[0]]
    t_ref_ctrl = t if ref_controls is not None else None

    labels = [
        ("x  [m]",       "z  [m]"),
        ("vx  [m/s]",    "vz  [m/s]"),
        ("θ  [rad]",     "ω  [rad/s]"),
        ("T  [N]",       "δ  [rad]"),
    ]
    state_idx = [(0, 1), (2, 3), (4, 5)]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    colors = ["#00e5ff", "#76ff03", "#ff9100", "#e040fb", "#ff5252", "#40c4ff"]

    for row in range(3):
        for col in range(2):
            ax = axes[row, col]
            si = state_idx[row][col]
            ax.plot(t, states[:, si], color=colors[row * 2 + col], lw=1.2)
            if ref_states is not None:
                ax.plot(t[: ref_states.shape[0]], ref_states[:, si],
                        ls="--", color="#aaaaaa", lw=0.9, label="ref")
            ax.set_ylabel(labels[row][col], fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="best")

    # T subplot
    ax_T = axes[3, 0]
    ax_T.plot(t_ctrl, controls[:, 0], color=colors[4], lw=1.2)
    if ref_controls is not None:
        ax_T.plot(t_ref_ctrl[: ref_controls.shape[0]], ref_controls[:, 0],
                  ls="--", color="#aaaaaa", lw=0.9, label="ref")
    ax_T.axhline(params.T_min, ls=":", color="#ff867c", lw=0.8)
    ax_T.axhline(params.T_max, ls=":", color="#ff867c", lw=0.8)
    ax_T.axhspan(params.T_min, params.T_max, color="#ff867c", alpha=0.07)
    ax_T.set_ylabel(labels[3][0], fontsize=9)
    ax_T.grid(True, alpha=0.3)

    # delta subplot
    ax_d = axes[3, 1]
    ax_d.plot(t_ctrl, controls[:, 1], color=colors[5], lw=1.2)
    if ref_controls is not None:
        ax_d.plot(t_ref_ctrl[: ref_controls.shape[0]], ref_controls[:, 1],
                  ls="--", color="#aaaaaa", lw=0.9, label="ref")
    ax_d.axhline(-params.delta_max, ls=":", color="#81d4fa", lw=0.8)
    ax_d.axhline(params.delta_max, ls=":", color="#81d4fa", lw=0.8)
    ax_d.axhspan(-params.delta_max, params.delta_max, color="#81d4fa", alpha=0.07)
    ax_d.set_ylabel(labels[3][1], fontsize=9)
    ax_d.grid(True, alpha=0.3)

    for col in range(2):
        axes[3, col].set_xlabel("time  [s]", fontsize=9)

    fig.tight_layout()
    return fig
