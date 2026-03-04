from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .integrators import rk4
from .plot_utils import set_3d_equal_aspect
from .rigid_body_dynamics import (
    make_rigid_body_ode,
    inertia_tensor_box,
    pack_state,
    unpack_state,
    quat_normalize,
    quat_to_rotation_matrix,
)


def show_tumbling_box(
    M: float = 5.0,
    dims: tuple[float, float, float] = (1.0, 2.0, 3.0),
    x0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
    T: float = 3.0,
    h: float = 0.002,
):
    lx, ly, lz = dims
    I_body = inertia_tensor_box(M, lx, ly, lz)
    g = 9.81

    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0])
    if v0 is None:
        v0 = np.array([2.0, 0.0, 12.0])
    if omega0 is None:
        omega0 = np.array([0.5, 4.0, 0.2])

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    P0 = M * v0
    L0 = I_body @ omega0

    state0 = pack_state(x0, q0, P0, L0)
    ode = make_rigid_body_ode(M, I_body)
    res = rk4(ode, state0, (0.0, T), h)

    # renormalize quaternions
    for i in range(len(res.t)):
        res.y[i, 3:7] = quat_normalize(res.y[i, 3:7])

    positions = res.y[:, 0:3]
    momenta_lin = res.y[:, 7:10]
    momenta_ang = res.y[:, 10:13]

    I_body_inv = np.linalg.inv(I_body)
    omegas_world = np.empty((len(res.t), 3))
    for i in range(len(res.t)):
        R = quat_to_rotation_matrix(res.y[i, 3:7])
        omegas_world[i] = R @ I_body_inv @ R.T @ momenta_ang[i]

    KE_trans = 0.5 * np.sum(momenta_lin**2, axis=1) / M
    KE_rot = np.array([
        0.5 * momenta_ang[i] @ np.linalg.inv(
            quat_to_rotation_matrix(res.y[i, 3:7]) @ I_body @
            quat_to_rotation_matrix(res.y[i, 3:7]).T
        ) @ momenta_ang[i]
        for i in range(len(res.t))
    ])
    PE = M * g * positions[:, 2]
    E_total = KE_trans + KE_rot + PE
    L_mag = np.linalg.norm(momenta_ang, axis=1)

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # --- panel 1: 3D trajectory with body axes ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
             "k-", lw=1.0, alpha=0.5, label="CM trajectory")

    axis_len = max(dims) * 0.5
    axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    n_arrows = 12
    indices = np.linspace(0, len(res.t) - 1, n_arrows, dtype=int)
    for idx in indices:
        R = quat_to_rotation_matrix(res.y[idx, 3:7])
        p = positions[idx]
        for j, c in enumerate(axis_colors):
            d = R[:, j] * axis_len
            ax1.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                       color=c, arrow_length_ratio=0.15, lw=1.2, alpha=0.7)

    xs = np.concatenate([positions[:, 0], [positions[:, 0].min() - axis_len, positions[:, 0].max() + axis_len]])
    ys_ext = np.concatenate([positions[:, 1], [positions[:, 1].min() - axis_len, positions[:, 1].max() + axis_len]])
    zs_ext = np.concatenate([positions[:, 2], [positions[:, 2].min() - axis_len, positions[:, 2].max() + axis_len]])
    set_3d_equal_aspect(ax1, xs, ys_ext, zs_ext, margin=0.1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Tumbling box: CM trajectory + body axes")
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=c, lw=2, label=l)
                      for c, l in zip(axis_colors, ["body x", "body y", "body z"])]
    ax1.legend(handles=legend_handles, fontsize=7, loc="upper left")

    # --- panel 2: angular velocity ---
    ax2 = fig.add_subplot(gs[0, 1])
    labels_w = ["ωx", "ωy", "ωz"]
    for j in range(3):
        ax2.plot(res.t, omegas_world[:, j], color=axis_colors[j], lw=1.0, label=labels_w[j])
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("ω [rad/s]")
    ax2.set_title("Angular velocity (world frame)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- panel 3: energy conservation ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(res.t, KE_trans, "--", color="#e74c3c", lw=0.8, label="KE trans")
    ax3.plot(res.t, KE_rot, "--", color="#3498db", lw=0.8, label="KE rot")
    ax3.plot(res.t, PE, "--", color="#2ecc71", lw=0.8, label="PE")
    ax3.plot(res.t, E_total, "k-", lw=1.5, label="Total E")
    rel_drift = abs(E_total[-1] / E_total[0] - 1) if E_total[0] != 0 else 0.0
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("Energy [J]")
    ax3.set_title(f"Energy conservation (drift: {rel_drift:.2e})")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- panel 4: |L| conservation ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(res.t, L_mag, "k-", lw=1.2)
    rel_drift_L = abs(L_mag[-1] / L_mag[0] - 1) if L_mag[0] != 0 else 0.0
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("|L| [kg·m²/s]")
    ax4.set_title(f"|L| conservation (drift: {rel_drift_L:.2e})")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"Rigid body simulation: box {lx}×{ly}×{lz} m, M={M} kg, RK4 h={h}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    plt.show()
    return fig
