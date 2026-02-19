from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, FloatSlider


def _simulate_block_on_incline(
    alpha_deg: float, mu_s: float, mu_k: float,
    m: float = 1.0, g: float = 9.81,
    dt: float = 0.001, T: float = 5.0,
    friction_type: str = "discontinuous",
    eps: float = 1e-3,
):
    alpha = np.radians(alpha_deg)
    n_steps = int(T / dt)
    xs = np.empty(n_steps)
    vs = np.empty(n_steps)
    Ffs = np.empty(n_steps)
    ts = np.arange(n_steps) * dt

    x, v = 0.0, 0.0
    F_gravity_tangent = m * g * np.sin(alpha)
    F_normal = m * g * np.cos(alpha)

    for i in range(n_steps):
        xs[i] = x
        vs[i] = v

        if friction_type == "discontinuous":
            if abs(v) < 1e-10 and abs(F_gravity_tangent) <= mu_s * F_normal:
                F_friction = -F_gravity_tangent
            else:
                F_friction = -mu_k * F_normal * np.sign(v) if abs(v) > 1e-10 else 0.0
        elif friction_type == "regularized":
            F_friction = -mu_k * F_normal * v / max(abs(v), eps)
        elif friction_type == "tanh":
            F_friction = -mu_k * F_normal * np.tanh(v / eps)
        else:
            F_friction = 0.0

        Ffs[i] = F_friction
        a = (F_gravity_tangent + F_friction) / m
        v += a * dt
        x += v * dt

        if friction_type == "discontinuous":
            if abs(v) < 1e-8 and abs(F_gravity_tangent) <= mu_s * F_normal:
                v = 0.0

    return ts, xs, vs, Ffs


def show_friction_comparison(
    alpha_deg: float = 35.0, mu: float = 0.5,
    fig=None,
):
    if fig is None:
        fig = plt.figure(figsize=(15, 8))
    else:
        fig.clf()
    gs = GridSpec(2, 2, figure=fig)

    ax_x = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_F = fig.add_subplot(gs[1, 0])
    ax_model = fig.add_subplot(gs[1, 1])

    mu_s, mu_k = mu, mu * 0.8

    styles = [
        ("Discontinuous Coulomb", "discontinuous", "#e74c3c"),
        ("Regularized (ε=0.01)",  "regularized",   "#2ecc71"),
        ("Tanh (ε=0.01)",         "tanh",           "#3498db"),
    ]

    for name, ftype, color in styles:
        ts, xs, vs, Ffs = _simulate_block_on_incline(
            alpha_deg, mu_s, mu_k, friction_type=ftype, eps=0.01,
        )
        ax_x.plot(ts, xs, color=color, lw=1.2, label=name)
        ax_v.plot(ts, vs, color=color, lw=1.2, label=name)
        ax_F.plot(ts, Ffs, color=color, lw=1.0, label=name)

    for ax in [ax_x, ax_v, ax_F]:
        ax.set_xlabel("t")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax_x.set_ylabel("x (along incline)")
    ax_x.set_title(f"Position  (α={alpha_deg}°, μₛ={mu_s}, μₖ={mu_k})")
    ax_v.set_ylabel("v")
    ax_v.set_title("Velocity")
    ax_F.set_ylabel("F_friction")
    ax_F.set_title("Friction force")

    # friction force vs velocity curve
    v_range = np.linspace(-1, 1, 1000)
    F_normal = 9.81
    eps = 0.01

    f_disc = np.where(np.abs(v_range) < 1e-10, 0.0, -mu_k * F_normal * np.sign(v_range))
    f_reg = -mu_k * F_normal * v_range / np.maximum(np.abs(v_range), eps)
    f_tanh = -mu_k * F_normal * np.tanh(v_range / eps)

    ax_model.plot(v_range, f_disc, color="#e74c3c", lw=1.5, label="Discontinuous")
    ax_model.plot(v_range, f_reg, color="#2ecc71", lw=1.5, label="Regularized")
    ax_model.plot(v_range, f_tanh, color="#3498db", lw=1.5, label="Tanh")
    ax_model.axhline(0, color="k", lw=0.5)
    ax_model.axvline(0, color="k", lw=0.5)
    ax_model.set_xlabel("v")
    ax_model.set_ylabel("F_friction")
    ax_model.set_title("Friction models: F(v)")
    ax_model.legend(fontsize=8)
    ax_model.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_static_vs_kinetic(mu: float = 0.5):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    angles = [15.0, 25.0, 40.0]
    critical_angle = np.degrees(np.arctan(mu))

    for ax, alpha_deg in zip(axes, angles):
        ts, xs, vs, Ffs = _simulate_block_on_incline(
            alpha_deg, mu, mu * 0.8, friction_type="regularized", eps=0.01,
        )
        ax.plot(ts, vs, "k-", lw=1.2)
        ax.set_xlabel("t")
        ax.set_ylabel("v")
        regime = "static" if alpha_deg < critical_angle else "sliding"
        ax.set_title(f"α = {alpha_deg}° ({regime}),  α_crit = {critical_angle:.1f}°")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Block on incline:  μ = {mu}", y=1.02)
    fig.tight_layout()
    plt.show()


_friction_interactive_fig = [None]


def show_friction_interactive():
    @interact(
        alpha_deg=FloatSlider(min=5, max=60, step=1, value=35, description="α (°)"),
        mu=FloatSlider(min=0.1, max=1.5, step=0.05, value=0.5, description="μ"),
    )
    def _update(alpha_deg, mu):
        if _friction_interactive_fig[0] is None:
            _friction_interactive_fig[0] = plt.figure(figsize=(15, 8))
        show_friction_comparison(alpha_deg=alpha_deg, mu=mu, fig=_friction_interactive_fig[0])


def show_friction_cone(mu: float = 0.5, n_pyramid: int = 8):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from .plot_utils import set_3d_equal_aspect

    fig = plt.figure(figsize=(14, 6))

    # --- left: cone with example forces ---
    ax1 = fig.add_subplot(121, projection="3d")

    Fn_max = 1.0
    theta = np.linspace(0, 2 * np.pi, 80)
    Fn_line = np.linspace(0, Fn_max, 40)
    TH, FN = np.meshgrid(theta, Fn_line)
    Ft1 = mu * FN * np.cos(TH)
    Ft2 = mu * FN * np.sin(TH)

    ax1.plot_surface(Ft1, Ft2, FN, alpha=0.15, color="#3498db", edgecolor="none")

    # cone boundary at Fn = Fn_max
    ax1.plot(mu * Fn_max * np.cos(theta), mu * Fn_max * np.sin(theta),
             np.full_like(theta, Fn_max), color="#3498db", lw=1.5, alpha=0.6)

    # valid force (inside cone)
    F_valid = np.array([0.1, 0.15, 0.8])
    ax1.quiver(0, 0, 0, *F_valid, color="#2ecc71", arrow_length_ratio=0.1, lw=2.5)
    ax1.text(F_valid[0] + 0.02, F_valid[1] + 0.02, F_valid[2], "valid",
             fontsize=10, color="#2ecc71", weight="bold")

    # invalid force (outside cone)
    F_invalid = np.array([0.4, 0.3, 0.5])
    ax1.quiver(0, 0, 0, *F_invalid, color="#e74c3c", arrow_length_ratio=0.1, lw=2.5)
    ax1.text(F_invalid[0] + 0.02, F_invalid[1] + 0.02, F_invalid[2], "invalid (slip)",
             fontsize=10, color="#e74c3c", weight="bold")

    r_lim = mu * Fn_max * 1.3
    xs = np.array([-r_lim, r_lim, 0, F_valid[0], F_invalid[0]])
    ys = np.array([-r_lim, r_lim, 0, F_valid[1], F_invalid[1]])
    zs = np.array([0, Fn_max * 1.1, F_valid[2], F_invalid[2]])
    set_3d_equal_aspect(ax1, xs, ys, zs, margin=0.15)
    ax1.set_xlabel("$F_{t1}$", fontsize=11)
    ax1.set_ylabel("$F_{t2}$", fontsize=11)
    ax1.set_zlabel("$F_n$", fontsize=11)
    ax1.set_title(f"Coulomb friction cone  ($\\mu = {mu}$)", fontsize=11)

    # --- right: cone + pyramid overlay ---
    ax2 = fig.add_subplot(122, projection="3d")

    ax2.plot_surface(Ft1, Ft2, FN, alpha=0.1, color="#3498db", edgecolor="none")
    ax2.plot(mu * Fn_max * np.cos(theta), mu * Fn_max * np.sin(theta),
             np.full_like(theta, Fn_max), color="#3498db", lw=1.0, alpha=0.4)

    # friction pyramid
    angles_pyr = np.linspace(0, 2 * np.pi, n_pyramid, endpoint=False)
    cos_corr = np.cos(np.pi / n_pyramid)  # inscribed correction
    verts_top = np.array([
        [mu * Fn_max * cos_corr * np.cos(a),
         mu * Fn_max * cos_corr * np.sin(a),
         Fn_max]
        for a in angles_pyr
    ])
    apex = np.array([0.0, 0.0, 0.0])

    # side faces
    faces = []
    for i in range(n_pyramid):
        j = (i + 1) % n_pyramid
        faces.append([apex, verts_top[i], verts_top[j]])

    # top face
    faces.append(list(verts_top))

    poly = Poly3DCollection(faces, alpha=0.25, facecolor="#e67e22", edgecolor="#c0392b", lw=0.8)
    ax2.add_collection3d(poly)

    # label vertices
    for i, v in enumerate(verts_top):
        ax2.plot([v[0]], [v[1]], [v[2]], "o", color="#c0392b", markersize=3)

    set_3d_equal_aspect(ax2, xs, ys, zs, margin=0.15)
    ax2.set_xlabel("$F_{t1}$", fontsize=11)
    ax2.set_ylabel("$F_{t2}$", fontsize=11)
    ax2.set_zlabel("$F_n$", fontsize=11)
    ax2.set_title(f"Friction pyramid ({n_pyramid} edges) inscribed in cone", fontsize=11)

    fig.tight_layout()
    plt.show()
