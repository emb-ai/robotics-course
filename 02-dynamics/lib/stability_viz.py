from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _amplification_explicit_euler(z: np.ndarray) -> np.ndarray:
    return np.abs(1 + z)


def _amplification_rk2(z: np.ndarray) -> np.ndarray:
    return np.abs(1 + z + z**2 / 2)


def _amplification_rk4(z: np.ndarray) -> np.ndarray:
    return np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)


def _amplification_implicit_euler(z: np.ndarray) -> np.ndarray:
    return np.abs(1 / (1 - z))


def show_stability_regions(
    xlim: tuple[float, float] = (-5, 2),
    ylim: tuple[float, float] = (-3.5, 3.5),
    resolution: int = 500,
):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    methods = [
        ("Explicit Euler", _amplification_explicit_euler),
        ("RK2 (Midpoint)", _amplification_rk2),
        ("RK4 (Classic)", _amplification_rk4),
        ("Implicit Euler", _amplification_implicit_euler),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (name, amp_fn) in zip(axes, methods):
        amp = amp_fn(Z)
        stable = amp <= 1.0

        ax.contourf(X, Y, stable.astype(float), levels=[0.5, 1.5], colors=["#2ecc71"], alpha=0.3)
        ax.contour(X, Y, amp, levels=[1.0], colors=["#2c3e50"], linewidths=1.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Re(hλ)")
        ax.set_ylabel("Im(hλ)")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Stability regions  (|g(hλ)| ≤ 1 shaded)", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.show()


def show_stability_regions_overlay(
    xlim: tuple[float, float] = (-5, 2),
    ylim: tuple[float, float] = (-3.5, 3.5),
    resolution: int = 500,
):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    methods = [
        ("Explicit Euler", _amplification_explicit_euler, "#e74c3c"),
        ("RK2", _amplification_rk2, "#f39c12"),
        ("RK4", _amplification_rk4, "#3498db"),
    ]

    for name, amp_fn, color in methods:
        amp = amp_fn(Z)
        ax.contour(X, Y, amp, levels=[1.0], colors=[color], linewidths=2)
        ax.contourf(X, Y, (amp <= 1.0).astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.1)

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Re(hλ)")
    ax.set_ylabel("Im(hλ)")
    ax.set_title("Stability regions: explicit methods compared")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # manual legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=n) for n, _, c in methods]
    handles.append(Line2D([0], [0], color="gray", lw=1, ls="--", label="Implicit Euler: entire left half-plane"))
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.show()


def show_stability_with_examples():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    resolution = 400
    x = np.linspace(-5, 2, resolution)
    y = np.linspace(-3.5, 3.5, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # left: explicit Euler with marked eigenvalues
    ax = axes[0]
    amp = _amplification_explicit_euler(Z)
    ax.contourf(X, Y, (amp <= 1.0).astype(float), levels=[0.5, 1.5], colors=["#2ecc71"], alpha=0.25)
    ax.contour(X, Y, amp, levels=[1.0], colors=["#2c3e50"], linewidths=1.5)

    examples = [
        (-1.0, 0, "h=0.1, λ=-10", "o", "#2ecc71"),
        (-2.5, 0, "h=0.1, λ=-25\n(UNSTABLE!)", "x", "#e74c3c"),
        (0, 1.0, "h=0.1, ω=10\n(oscillator)", "s", "#3498db"),
        (0, 2.5, "h=0.1, ω=25\n(UNSTABLE!)", "D", "#e74c3c"),
    ]
    for re, im, label, marker, color in examples:
        ax.plot(re, im, marker, color=color, markersize=10, markeredgewidth=2)
        ax.annotate(label, (re, im), textcoords="offset points", xytext=(10, 10), fontsize=7, color=color)

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Re(hλ)")
    ax.set_ylabel("Im(hλ)")
    ax.set_title("Explicit Euler: stability region + examples")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # right: amplification factor along imaginary axis (pure oscillator)
    ax2 = axes[1]
    h_omega_values = np.linspace(0, 3, 200)

    for name, amp_fn, color in [
        ("Explicit Euler", _amplification_explicit_euler, "#e74c3c"),
        ("RK2", _amplification_rk2, "#f39c12"),
        ("RK4", _amplification_rk4, "#3498db"),
        ("Implicit Euler", _amplification_implicit_euler, "#9b59b6"),
    ]:
        amp = amp_fn(1j * h_omega_values)
        ax2.plot(h_omega_values, amp, color=color, lw=2, label=name)

    ax2.axhline(1.0, color="k", lw=1, ls=":")
    ax2.set_xlabel("h·ω  (imaginary axis)")
    ax2.set_ylabel("|g(ihω)|")
    ax2.set_title("Amplification factor for pure oscillator (λ = iω)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.5)

    fig.tight_layout()
    plt.show()
