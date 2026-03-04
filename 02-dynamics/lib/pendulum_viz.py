from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, FancyArrowPatch
from ipywidgets import interact, FloatSlider

from .integrators import explicit_euler, symplectic_euler, rk4, implicit_euler


def _pendulum_rhs(g_over_L: float):
    def f(t, y):
        theta, omega = y
        return np.array([omega, -g_over_L * np.sin(theta)])
    return f


def _pendulum_jac(g_over_L: float):
    def jac(t, y):
        theta = y[0]
        return np.array([[0.0, 1.0], [-g_over_L * np.cos(theta), 0.0]])
    return jac


def _pendulum_accel(g_over_L: float):
    def a(t, x, v):
        return -g_over_L * np.sin(x)
    return a


def _pendulum_energy(theta, omega, g_over_L):
    return 0.5 * omega**2 - g_over_L * np.cos(theta)


def _energy_contours(ax, g_over_L: float, theta_range=(-np.pi, np.pi), n_levels: int = 20):
    th = np.linspace(theta_range[0], theta_range[1], 400)
    om = np.linspace(-8, 8, 400)
    TH, OM = np.meshgrid(th, om)
    E = _pendulum_energy(TH, OM, g_over_L)

    E_sep = g_over_L  # separatrix energy
    levels_below = np.linspace(-g_over_L * 1.05, E_sep - 0.01, n_levels // 2)
    levels_above = np.linspace(E_sep + 0.2, E_sep + 15, n_levels // 2)

    ax.contour(TH, OM, E, levels=levels_below, colors="#3498db", linewidths=0.6, alpha=0.5)
    ax.contour(TH, OM, E, levels=[E_sep], colors="#e74c3c", linewidths=1.5, alpha=0.8)
    ax.contour(TH, OM, E, levels=levels_above, colors="#95a5a6", linewidths=0.4, alpha=0.4)


# ---------------------------------------------------------------------------
# 1. Pendulum diagram
# ---------------------------------------------------------------------------

def show_pendulum_diagram(L: float = 1.0, theta_deg: float = 30.0):
    theta = np.radians(theta_deg)
    bob_x = L * np.sin(theta)
    bob_y = -L * np.cos(theta)

    fig, ax = plt.subplots(figsize=(5, 5.5))

    # pivot
    ax.plot(0, 0, "ks", markersize=10, zorder=5)

    # rod
    ax.plot([0, bob_x], [0, bob_y], "k-", lw=2.5, zorder=3)

    # bob
    ax.plot(bob_x, bob_y, "o", color="#3498db", markersize=22, markeredgecolor="k",
            markeredgewidth=1.5, zorder=4)

    # vertical reference (dashed)
    ax.plot([0, 0], [0, -L * 1.15], "k--", lw=0.8, alpha=0.4)

    # angle arc
    arc = Arc((0, 0), L * 0.5, L * 0.5, angle=0,
              theta1=270, theta2=270 + theta_deg, color="#e74c3c", lw=2)
    ax.add_patch(arc)
    label_r = L * 0.35
    label_angle = np.radians(270 + theta_deg / 2)
    ax.text(label_r * np.cos(label_angle), label_r * np.sin(label_angle),
            r"$\theta$", fontsize=16, color="#e74c3c", ha="center", va="center")

    # gravity arrow
    arr_start_y = bob_y
    arr_len = L * 0.35
    ax.annotate("", xy=(bob_x, arr_start_y - arr_len), xytext=(bob_x, arr_start_y),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                                color="#2c3e50", lw=2))
    ax.text(bob_x + 0.08, arr_start_y - arr_len / 2, r"$mg$",
            fontsize=14, color="#2c3e50", va="center")

    # tangential force component
    tang_dir = np.array([np.cos(theta), np.sin(theta)])
    F_tang = -np.sin(theta)  # sign shows direction
    tang_vec = tang_dir * F_tang * arr_len * 1.5
    ax.annotate("", xy=(bob_x + tang_vec[0], bob_y + tang_vec[1]),
                xytext=(bob_x, bob_y),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.12",
                                color="#27ae60", lw=2))
    ax.text(bob_x + tang_vec[0] - 0.15, bob_y + tang_vec[1] + 0.08,
            r"$-mg\sin\theta$", fontsize=12, color="#27ae60")

    # labels
    mid_x, mid_y = bob_x / 2, bob_y / 2
    ax.text(mid_x - 0.12, mid_y, r"$L$", fontsize=14, ha="right")
    ax.text(bob_x, bob_y - 0.15, r"$m$", fontsize=13, ha="center", va="top", color="#3498db")

    ax.set_xlim(-L * 0.8, L * 0.8)
    ax.set_ylim(-L * 1.5, L * 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Simple pendulum", fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Phase space (analytical energy contours)
# ---------------------------------------------------------------------------

def show_pendulum_phase_space(g_over_L: float = 9.81):
    fig, ax = plt.subplots(figsize=(10, 5))
    _energy_contours(ax, g_over_L)

    # mark some initial conditions
    ics = [
        (np.radians(30), 0, "small oscillation"),
        (np.radians(90), 0, "medium oscillation"),
        (np.radians(170), 0, "near separatrix"),
    ]
    for th0, om0, label in ics:
        ax.plot(th0, om0, "o", markersize=7, color="#2c3e50", zorder=5)
        ax.annotate(label, (th0, om0), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color="#2c3e50")

    # annotations
    ax.text(0, 0.4, "stable\nequilibrium", fontsize=9, ha="center", color="#3498db", alpha=0.7)
    ax.text(np.pi * 0.85, 6.5, "rotation\n(over the top)", fontsize=9, ha="center", color="#95a5a6")

    sep_E = g_over_L
    om_at_0 = np.sqrt(2 * (sep_E + g_over_L))
    ax.annotate("separatrix", xy=(0, om_at_0 * 0.97), xytext=(0.8, om_at_0 + 0.5),
                fontsize=9, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1))

    ax.set_xlabel(r"$\theta$", fontsize=13)
    ax.set_ylabel(r"$\omega$", fontsize=13)
    ax.set_title(r"Phase space of the pendulum — level curves of $E = \frac{1}{2}\omega^2 - \frac{g}{L}\cos\theta$",
                 fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-8, 8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Explicit Euler on the pendulum (shows divergence)
# ---------------------------------------------------------------------------

def show_pendulum_explicit_euler(
    theta0_deg: float = 30.0, h: float = 0.05, T: float = 20.0,
    g_over_L: float = 9.81,
):
    theta0 = np.radians(theta0_deg)
    y0 = np.array([theta0, 0.0])
    f = _pendulum_rhs(g_over_L)

    res = explicit_euler(f, y0, (0.0, T), h)
    ref = rk4(f, y0, (0.0, T), h / 20)

    fig = plt.figure(figsize=(15, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # theta(t)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ref.t, ref.y[:, 0], "k-", lw=1.5, alpha=0.3, label="Reference (RK4)")
    ax1.plot(res.t, res.y[:, 0], color="#e74c3c", lw=1.2, label="Explicit Euler")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\theta$")
    ax1.set_title(r"$\theta(t)$" + f"  (h = {h})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # phase portrait on top of energy contours
    ax2 = fig.add_subplot(gs[1])
    _energy_contours(ax2, g_over_L)
    ax2.plot(res.y[:, 0], res.y[:, 1], color="#e74c3c", lw=1.0, alpha=0.9, label="Explicit Euler")
    ax2.plot(theta0, 0, "ko", markersize=6, zorder=5)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\omega$")
    ax2.set_title("Phase portrait — spiraling outward")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # energy vs time
    ax3 = fig.add_subplot(gs[2])
    E = _pendulum_energy(res.y[:, 0], res.y[:, 1], g_over_L)
    E0 = _pendulum_energy(theta0, 0.0, g_over_L)
    E_ref = _pendulum_energy(ref.y[:, 0], ref.y[:, 1], g_over_L)
    ax3.plot(ref.t, E_ref, "k-", lw=1.5, alpha=0.3, label="Reference")
    ax3.plot(res.t, E, color="#e74c3c", lw=1.2, label="Explicit Euler")
    ax3.axhline(E0, color="k", ls=":", lw=0.8, alpha=0.4)
    ax3.set_xlabel("t")
    ax3.set_ylabel("E(t)")
    ax3.set_title("Energy — monotonically growing")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Symplectic Euler on the pendulum (shows success)
# ---------------------------------------------------------------------------

def show_pendulum_symplectic_euler(
    theta0_deg: float = 30.0, h: float = 0.05, T: float = 20.0,
    g_over_L: float = 9.81,
):
    theta0 = np.radians(theta0_deg)
    y0 = np.array([theta0, 0.0])
    f = _pendulum_rhs(g_over_L)
    accel = _pendulum_accel(g_over_L)

    ts, xs, vs = symplectic_euler(accel, np.array([theta0]), np.array([0.0]), (0.0, T), h)
    ref = rk4(f, y0, (0.0, T), h / 20)

    fig = plt.figure(figsize=(15, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # theta(t)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ref.t, ref.y[:, 0], "k-", lw=1.5, alpha=0.3, label="Reference (RK4)")
    ax1.plot(ts, xs[:, 0], color="#2ecc71", lw=1.2, label="Symplectic Euler")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\theta$")
    ax1.set_title(r"$\theta(t)$" + f"  (h = {h})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # phase portrait
    ax2 = fig.add_subplot(gs[1])
    _energy_contours(ax2, g_over_L)
    ax2.plot(xs[:, 0], vs[:, 0], color="#2ecc71", lw=1.0, alpha=0.9, label="Symplectic Euler")
    ax2.plot(theta0, 0, "ko", markersize=6, zorder=5)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\omega$")
    ax2.set_title("Phase portrait — stays on closed curve")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # energy vs time
    ax3 = fig.add_subplot(gs[2])
    E = _pendulum_energy(xs[:, 0], vs[:, 0], g_over_L)
    E0 = _pendulum_energy(theta0, 0.0, g_over_L)
    E_ref = _pendulum_energy(ref.y[:, 0], ref.y[:, 1], g_over_L)
    ax3.plot(ref.t, E_ref, "k-", lw=1.5, alpha=0.3, label="Reference")
    ax3.plot(ts, E, color="#2ecc71", lw=1.2, label="Symplectic Euler")
    ax3.axhline(E0, color="k", ls=":", lw=0.8, alpha=0.4)
    ax3.set_xlabel("t")
    ax3.set_ylabel("E(t)")
    ax3.set_title("Energy — bounded oscillation")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Side-by-side comparison (the payoff demo)
# ---------------------------------------------------------------------------

def show_pendulum_euler_vs_symplectic(
    theta0_deg: float = 30.0, h: float = 0.05, T: float = 20.0,
    g_over_L: float = 9.81,
):
    theta0 = np.radians(theta0_deg)
    y0 = np.array([theta0, 0.0])
    f = _pendulum_rhs(g_over_L)
    accel = _pendulum_accel(g_over_L)

    res_euler = explicit_euler(f, y0, (0.0, T), h / 2)
    ts_se, xs_se, vs_se = symplectic_euler(accel, np.array([theta0]), np.array([0.0]), (0.0, T), h)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    # phase portrait: explicit Euler
    ax1 = fig.add_subplot(gs[0, 0])
    _energy_contours(ax1, g_over_L)
    ax1.plot(res_euler.y[:, 0], res_euler.y[:, 1], color="#e74c3c", lw=1.0)
    ax1.plot(theta0, 0, "ko", markersize=6, zorder=5)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\omega$")
    ax1.set_title(f"Explicit Euler  (h = {h})")
    ax1.grid(True, alpha=0.2)

    # phase portrait: symplectic Euler
    ax2 = fig.add_subplot(gs[0, 1])
    _energy_contours(ax2, g_over_L)
    ax2.plot(xs_se[:, 0], vs_se[:, 0], color="#2ecc71", lw=1.0)
    ax2.plot(theta0, 0, "ko", markersize=6, zorder=5)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\omega$")
    ax2.set_title(f"Symplectic Euler  (h = {h})")
    ax2.grid(True, alpha=0.2)

    # sync axis limits
    all_theta = np.concatenate([res_euler.y[:, 0], xs_se[:, 0]])
    all_omega = np.concatenate([res_euler.y[:, 1], vs_se[:, 0]])
    pad = 0.3
    th_lim = (min(all_theta.min(), -np.pi) - pad, max(all_theta.max(), np.pi) + pad)
    om_lim = (all_omega.min() - pad, all_omega.max() + pad)
    for ax in [ax1, ax2]:
        ax.set_xlim(th_lim)
        ax.set_ylim(om_lim)

    # energy comparison
    ax3 = fig.add_subplot(gs[1, :])
    E_euler = _pendulum_energy(res_euler.y[:, 0], res_euler.y[:, 1], g_over_L)
    E_symp = _pendulum_energy(xs_se[:, 0], vs_se[:, 0], g_over_L)
    E0 = _pendulum_energy(theta0, 0.0, g_over_L)
    ax3.plot(res_euler.t, E_euler, color="#e74c3c", lw=1.2, label="Explicit Euler")
    ax3.plot(ts_se, E_symp, color="#2ecc71", lw=1.2, label="Symplectic Euler")
    ax3.axhline(E0, color="k", ls=":", lw=0.8, alpha=0.4, label="True energy")
    ax3.set_xlabel("t")
    ax3.set_ylabel("E(t)")
    ax3.set_title("Energy: explicit Euler drifts up, symplectic Euler stays bounded")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_pendulum_euler_vs_symplectic_interactive(
    theta0_deg: float = 30.0, T: float = 20.0, g_over_L: float = 9.81,
):
    @interact(
        h=FloatSlider(min=0.005, max=0.15, step=0.005, value=0.05, description="h"),
    )
    def _update(h):
        show_pendulum_euler_vs_symplectic(theta0_deg=theta0_deg, h=h, T=T, g_over_L=g_over_L)


# ---------------------------------------------------------------------------
# Original comparison functions (used later in the lecture)
# ---------------------------------------------------------------------------

def show_pendulum_comparison(
    theta0_deg: float = 170.0, T: float = 20.0, h: float = 0.01,
    g_over_L: float = 9.81,
):
    theta0 = np.radians(theta0_deg)
    y0 = np.array([theta0, 0.0])
    t_span = (0.0, T)

    f = _pendulum_rhs(g_over_L)
    jac = _pendulum_jac(g_over_L)
    accel = _pendulum_accel(g_over_L)

    res_euler = explicit_euler(f, y0, t_span, h)
    ts_se, xs_se, vs_se = symplectic_euler(
        accel, np.array([theta0]), np.array([0.0]), t_span, h,
    )
    res_rk4 = rk4(f, y0, t_span, h)
    res_impl = implicit_euler(f, y0, t_span, h, jac=jac)

    h_ref = h / 20
    ref = rk4(f, y0, t_span, h_ref)

    methods = [
        ("Explicit Euler",   res_euler.t,  res_euler.y[:, 0],  res_euler.y[:, 1], "#e74c3c"),
        ("Symplectic Euler", ts_se,         xs_se[:, 0],        vs_se[:, 0],       "#2ecc71"),
        ("RK4",              res_rk4.t,    res_rk4.y[:, 0],    res_rk4.y[:, 1],   "#3498db"),
        ("Implicit Euler",   res_impl.t,   res_impl.y[:, 0],   res_impl.y[:, 1],  "#9b59b6"),
    ]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax_theta = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_error = fig.add_subplot(gs[1, 1])

    ax_theta.plot(ref.t, ref.y[:, 0], "k-", lw=1.5, alpha=0.4, label="Reference")
    for name, t, theta, omega, color in methods:
        ax_theta.plot(t, theta, "--", lw=1.2, color=color, label=name)
    ax_theta.set_xlabel("t")
    ax_theta.set_ylabel("θ")
    ax_theta.set_title(f"θ(t)  (h = {h}, θ₀ = {theta0_deg}°)")
    ax_theta.legend(fontsize=7)
    ax_theta.grid(True, alpha=0.3)

    for name, t, theta, omega, color in methods:
        E = _pendulum_energy(theta, omega, g_over_L)
        ax_energy.plot(t, E, "--", lw=1.2, color=color, label=name)
    E0 = _pendulum_energy(theta0, 0.0, g_over_L)
    ax_energy.axhline(E0, color="k", ls=":", lw=1, alpha=0.5)
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("E(t)")
    ax_energy.set_title("Energy vs time")
    ax_energy.legend(fontsize=7)
    ax_energy.grid(True, alpha=0.3)

    for name, t, theta, omega, color in methods:
        ax_phase.plot(theta, omega, "--", lw=1.0, color=color, label=name)
    ax_phase.set_xlabel("θ")
    ax_phase.set_ylabel("ω")
    ax_phase.set_title("Phase portrait")
    ax_phase.legend(fontsize=7)
    ax_phase.grid(True, alpha=0.3)

    h_values = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    h_ref_conv = 0.0002
    ref_conv = rk4(f, y0, (0.0, T), h_ref_conv)

    for method_name, method_fn, color in [
        ("Explicit Euler", lambda hh: explicit_euler(f, y0, (0.0, T), hh), "#e74c3c"),
        ("RK4",            lambda hh: rk4(f, y0, (0.0, T), hh),            "#3498db"),
    ]:
        errors = []
        valid_hs = []
        for hh in h_values:
            try:
                res = method_fn(hh)
                n_ref = int(round(T / h_ref_conv))
                n_coarse = int(round(T / hh))
                ratio = max(1, n_ref // n_coarse)
                ref_at_coarse = ref_conv.y[::ratio][:n_coarse + 1]
                if len(ref_at_coarse) == len(res.y):
                    err = np.max(np.abs(res.y[:, 0] - ref_at_coarse[:, 0]))
                else:
                    err = np.abs(res.y[-1, 0] - ref_conv.y[-1, 0])
                if np.isfinite(err) and err > 0:
                    errors.append(err)
                    valid_hs.append(hh)
            except Exception:
                pass
        if errors:
            ax_error.loglog(valid_hs, errors, "o--", color=color, label=method_name)

    hs_line = np.array([1e-3, 1e-1])
    ax_error.loglog(hs_line, 0.5 * hs_line**1, ":", color="#aaa", lw=1, label="O(h)")
    ax_error.loglog(hs_line, 0.5 * hs_line**4, ":", color="#888", lw=1, label="O(h⁴)")
    ax_error.set_xlabel("h")
    ax_error.set_ylabel("max |θ - θ_ref|")
    ax_error.set_title("Convergence (error vs step size)")
    ax_error.legend(fontsize=7)
    ax_error.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


def show_pendulum_interactive(
    theta0_deg: float = 170.0, T: float = 20.0, g_over_L: float = 9.81,
):
    @interact(
        h=FloatSlider(min=0.001, max=0.1, step=0.001, value=0.01, description="h"),
    )
    def _update(h):
        show_pendulum_comparison(theta0_deg=theta0_deg, T=T, h=h, g_over_L=g_over_L)
