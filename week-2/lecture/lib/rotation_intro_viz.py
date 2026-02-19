from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle


def show_compound_pendulum_diagram(L: float = 1.0, theta_deg: float = 30.0):
    theta = np.radians(theta_deg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax, is_compound in zip(axes, [False, True]):
        bob_x = L * np.sin(theta)
        bob_y = -L * np.cos(theta)
        cm_x = (L / 2) * np.sin(theta)
        cm_y = -(L / 2) * np.cos(theta)

        # pivot
        ax.plot(0, 0, "ks", markersize=10, zorder=5)

        # vertical reference
        ax.plot([0, 0], [0, -L * 1.15], "k--", lw=0.8, alpha=0.4)

        # angle arc
        arc = Arc((0, 0), L * 0.45, L * 0.45, angle=0,
                  theta1=270, theta2=270 + theta_deg, color="#e74c3c", lw=2)
        ax.add_patch(arc)
        label_r = L * 0.3
        label_angle = np.radians(270 + theta_deg / 2)
        ax.text(label_r * np.cos(label_angle), label_r * np.sin(label_angle),
                r"$\theta$", fontsize=15, color="#e74c3c", ha="center", va="center")

        if not is_compound:
            # simple pendulum: thin rod + point mass
            ax.plot([0, bob_x], [0, bob_y], "k-", lw=2, zorder=3)
            ax.plot(bob_x, bob_y, "o", color="#3498db", markersize=22,
                    markeredgecolor="k", markeredgewidth=1.5, zorder=4)

            # gravity at bob
            arr_len = L * 0.3
            ax.annotate("", xy=(bob_x, bob_y - arr_len), xytext=(bob_x, bob_y),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.12",
                                        color="#2c3e50", lw=2))
            ax.text(bob_x + 0.1, bob_y - arr_len / 2, r"$mg$", fontsize=13, color="#2c3e50")

            # lever arm brace
            ax.annotate("", xy=(0, 0), xytext=(bob_x, bob_y),
                        arrowprops=dict(arrowstyle="<->", color="#9b59b6", lw=1.5, ls="--"))
            ax.text(bob_x / 2 - 0.15, bob_y / 2 + 0.05, r"$L$", fontsize=14, color="#9b59b6")

            # info box
            info = (r"$I = mL^2$" + "\n"
                    + r"$\tau = -mgL\sin\theta$" + "\n"
                    + r"$\ddot\theta = -\frac{g}{L}\sin\theta$")
            ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=11,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2f8", ec="#aed6f1"))
            ax.set_title("Simple pendulum (point mass)", fontsize=12)

        else:
            # compound pendulum: thick rod
            rod_width = 0.06
            dx = rod_width * np.cos(theta)
            dy = rod_width * np.sin(theta)
            rod_corners = np.array([
                [-dx / 2, dy / 2],
                [dx / 2, -dy / 2],
                [bob_x + dx / 2, bob_y - dy / 2],
                [bob_x - dx / 2, bob_y + dy / 2],
            ])
            from matplotlib.patches import Polygon
            rod = Polygon(rod_corners, closed=True, fc="#3498db", ec="k", lw=1.5, zorder=3, alpha=0.8)
            ax.add_patch(rod)

            # center of mass marker
            ax.plot(cm_x, cm_y, "o", color="#e67e22", markersize=10,
                    markeredgecolor="k", markeredgewidth=1.2, zorder=5)
            ax.text(cm_x + 0.1, cm_y + 0.05, "CM", fontsize=11, color="#e67e22", weight="bold")

            # gravity arrow at CM
            arr_len = L * 0.3
            ax.annotate("", xy=(cm_x, cm_y - arr_len), xytext=(cm_x, cm_y),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.12",
                                        color="#2c3e50", lw=2))
            ax.text(cm_x + 0.1, cm_y - arr_len / 2, r"$Mg$", fontsize=13, color="#2c3e50")

            # lever arm to CM
            ax.annotate("", xy=(0, 0), xytext=(cm_x, cm_y),
                        arrowprops=dict(arrowstyle="<->", color="#9b59b6", lw=1.5, ls="--"))
            ax.text(cm_x / 2 - 0.18, cm_y / 2 + 0.05, r"$\frac{L}{2}$",
                    fontsize=14, color="#9b59b6")

            # full length label
            ax.text(bob_x / 2 + 0.12, bob_y / 2, r"$L$", fontsize=13, color="#555")

            info = (r"$I = \frac{1}{3}ML^2$" + "\n"
                    + r"$\tau = -Mg\frac{L}{2}\sin\theta$" + "\n"
                    + r"$\ddot\theta = -\frac{3g}{2L}\sin\theta$")
            ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=11,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7", ec="#f9e79f"))
            ax.set_title("Compound pendulum (rigid rod)", fontsize=12)

        ax.set_xlim(-L * 0.6, L * 0.75)
        ax.set_ylim(-L * 1.45, L * 0.25)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def _energy_contours_generic(ax, omega0_sq: float, color: str, label: str,
                              theta_range=(-np.pi, np.pi), alpha: float = 0.5):
    th = np.linspace(theta_range[0], theta_range[1], 400)
    om = np.linspace(-8, 8, 400)
    TH, OM = np.meshgrid(th, om)
    E = 0.5 * OM**2 - omega0_sq * np.cos(TH)

    E_sep = omega0_sq
    levels_below = np.linspace(-omega0_sq * 1.05, E_sep - 0.01, 12)
    ax.contour(TH, OM, E, levels=levels_below, colors=color, linewidths=0.7, alpha=alpha)
    ax.contour(TH, OM, E, levels=[E_sep], colors=color, linewidths=1.8, alpha=min(alpha + 0.2, 1.0))


def show_simple_vs_compound_phase_space(g: float = 9.81, L: float = 1.0):
    omega0_sq_simple = g / L
    omega0_sq_compound = 3 * g / (2 * L)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # simple pendulum alone
    _energy_contours_generic(axes[0], omega0_sq_simple, "#3498db", "Simple", alpha=0.6)
    axes[0].set_title(r"Simple:  $\omega_0^2 = g/L$" + f" = {omega0_sq_simple:.2f}", fontsize=11)

    # compound pendulum alone
    _energy_contours_generic(axes[1], omega0_sq_compound, "#e67e22", "Compound", alpha=0.6)
    axes[1].set_title(r"Compound:  $\omega_0^2 = 3g/2L$" + f" = {omega0_sq_compound:.2f}", fontsize=11)

    # overlay
    _energy_contours_generic(axes[2], omega0_sq_simple, "#3498db", "Simple", alpha=0.45)
    _energy_contours_generic(axes[2], omega0_sq_compound, "#e67e22", "Compound", alpha=0.45)
    axes[2].set_title("Overlay: different $I$ shifts the separatrix", fontsize=11)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#3498db", lw=2, label=r"Simple ($I=mL^2$)"),
        Line2D([0], [0], color="#e67e22", lw=2, label=r"Compound ($I=\frac{1}{3}ML^2$)"),
    ]
    axes[2].legend(handles=handles, fontsize=9, loc="upper left")

    for ax in axes:
        ax.set_xlabel(r"$\theta$", fontsize=12)
        ax.set_ylabel(r"$\omega$", fontsize=12)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.show()
