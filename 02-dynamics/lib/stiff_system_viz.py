from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ipywidgets import interact, FloatSlider

from .integrators import explicit_euler, symplectic_euler, rk4, implicit_euler, bdf2


def _coupled_oscillators_rhs(k_stiff: float, k_soft: float, m: float = 1.0):
    """Two masses on a line: x1 attached to wall by stiff spring, x2 by soft spring, coupled."""
    def f(t, y):
        x1, x2, v1, v2 = y
        a1 = (-k_stiff * x1 + k_soft * (x2 - x1)) / m
        a2 = (-k_soft * (x2 - x1)) / m
        return np.array([v1, v2, a1, a2])
    return f


def _coupled_oscillators_jac(k_stiff: float, k_soft: float, m: float = 1.0):
    J = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [(-k_stiff - k_soft) / m, k_soft / m, 0, 0],
        [k_soft / m, -k_soft / m, 0, 0],
    ])
    def jac(t, y):
        return J
    return jac


def _coupled_accel(k_stiff: float, k_soft: float, m: float = 1.0):
    def a(t, x, v):
        x1, x2 = x
        a1 = (-k_stiff * x1 + k_soft * (x2 - x1)) / m
        a2 = (-k_soft * (x2 - x1)) / m
        return np.array([a1, a2])
    return a


def show_stiff_system_comparison(
    k_stiff: float = 1e4, k_soft: float = 1.0,
    T: float = 10.0, h: float = 0.01,
):
    f = _coupled_oscillators_rhs(k_stiff, k_soft)
    jac = _coupled_oscillators_jac(k_stiff, k_soft)
    accel = _coupled_accel(k_stiff, k_soft)
    y0 = np.array([1.0, 1.0, 0.0, 0.0])
    t_span = (0.0, T)

    eigenvalues = np.linalg.eigvals(jac(0, y0))
    max_eig = np.max(np.abs(eigenvalues))
    h_crit = 2.0 / max_eig

    results = {}
    timings = {}

    # reference: RK4 with tiny step
    h_ref = min(h_crit * 0.5, 0.0001)
    t0 = time.perf_counter()
    ref = rk4(f, y0, t_span, h_ref)
    timings["RK4 ref"] = time.perf_counter() - t0

    methods = [
        ("Explicit Euler", lambda: explicit_euler(f, y0, t_span, h)),
        ("RK4", lambda: rk4(f, y0, t_span, h)),
        ("Implicit Euler", lambda: implicit_euler(f, y0, t_span, h, jac=jac)),
        ("BDF2", lambda: bdf2(f, y0, t_span, h, jac=jac)),
    ]

    for name, run in methods:
        t0 = time.perf_counter()
        try:
            res = run()
            elapsed = time.perf_counter() - t0
            blew_up = np.any(np.abs(res.y) > 1e6)
            results[name] = res
            timings[name] = elapsed
            if blew_up:
                results[name] = None
        except Exception:
            results[name] = None
            timings[name] = 0.0

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    ax_x1 = fig.add_subplot(gs[0, 0])
    ax_x2 = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 0])
    ax_energy = fig.add_subplot(gs[1, 1])

    # subsample reference
    ref_skip = max(1, len(ref.t) // 2000)
    ax_x1.plot(ref.t[::ref_skip], ref.y[::ref_skip, 0], "k-", lw=1, alpha=0.3, label="Reference")
    ax_x2.plot(ref.t[::ref_skip], ref.y[::ref_skip, 1], "k-", lw=1, alpha=0.3, label="Reference")

    colors = {"Explicit Euler": "#e74c3c", "RK4": "#3498db", "Implicit Euler": "#9b59b6", "BDF2": "#2ecc71"}

    for name, color in colors.items():
        res = results.get(name)
        if res is not None:
            ax_x1.plot(res.t, res.y[:, 0], "--", color=color, lw=1.2, label=name)
            ax_x2.plot(res.t, res.y[:, 1], "--", color=color, lw=1.2, label=name)

            E = 0.5 * (res.y[:, 2]**2 + res.y[:, 3]**2) + \
                0.5 * k_stiff * res.y[:, 0]**2 + \
                0.5 * k_soft * (res.y[:, 1] - res.y[:, 0])**2
            ax_energy.plot(res.t, E, "--", color=color, lw=1.2, label=name)
        else:
            ax_x1.text(0.5, 0.5, f"{name}: DIVERGED", transform=ax_x1.transAxes,
                      fontsize=9, color=color, ha="center", alpha=0.7)

    for ax, title in [(ax_x1, "x₁ (stiff DOF)"), (ax_x2, "x₂ (soft DOF)")]:
        ax.set_xlabel("t")
        ax.set_ylabel(title.split()[0])
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("E")
    ax_energy.set_title("Total energy")
    ax_energy.legend(fontsize=7)
    ax_energy.grid(True, alpha=0.3)

    # info text
    ax_info.axis("off")
    info_lines = [
        f"k_stiff = {k_stiff:.0f},  k_soft = {k_soft:.1f}",
        f"Eigenvalues of Jacobian: {', '.join(f'{e:.1f}' for e in np.sort(eigenvalues)[::-1])}",
        f"Max |λ| = {max_eig:.1f}",
        f"Critical h for explicit methods: h < {h_crit:.6f}",
        f"Chosen h = {h}  {'(STABLE)' if h < h_crit else '(UNSTABLE for explicit!)'}",
        "",
        "Timings:",
    ]
    for name in ["Explicit Euler", "RK4", "Implicit Euler", "BDF2"]:
        status = "diverged" if results.get(name) is None else f"{timings.get(name, 0)*1000:.1f} ms"
        info_lines.append(f"  {name}: {status}")
    info_lines.append(f"  Reference (RK4, h={h_ref:.6f}): {timings.get('RK4 ref', 0)*1000:.1f} ms")

    ax_info.text(0.05, 0.95, "\n".join(info_lines), transform=ax_info.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    fig.suptitle(f"Stiff system: coupled oscillators (h = {h})", fontsize=13)
    fig.tight_layout()
    plt.show()


def show_stiff_explicit_euler_h_sweep(
    k_stiff: float = 1e4, k_soft: float = 1.0, T: float = 2.0,
):
    f = _coupled_oscillators_rhs(k_stiff, k_soft)
    jac = _coupled_oscillators_jac(k_stiff, k_soft)
    y0 = np.array([1.0, 1.0, 0.0, 0.0])

    eigenvalues = np.linalg.eigvals(jac(0, y0))
    max_eig = np.max(np.abs(eigenvalues))
    h_crit = 2.0 / max_eig

    h_values = [h_crit * 2, h_crit * 1.1, h_crit * 0.9, h_crit * 0.5]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    for ax, h in zip(axes, h_values):
        res = explicit_euler(f, y0, (0.0, T), h)
        stable = not np.any(np.abs(res.y) > 1e6)
        if stable:
            ax.plot(res.t, res.y[:, 0], "#2ecc71", lw=1.2)
            ax.plot(res.t, res.y[:, 1], "#3498db", lw=1.2)
        else:
            n_ok = np.argmax(np.abs(res.y[:, 0]) > 1e6)
            if n_ok == 0:
                n_ok = 10
            ax.plot(res.t[:n_ok], res.y[:n_ok, 0], "#e74c3c", lw=1.2)
            ax.text(0.5, 0.5, "DIVERGED", transform=ax.transAxes,
                   fontsize=14, color="#e74c3c", ha="center", weight="bold")

        label = "STABLE" if stable else "UNSTABLE"
        ax.set_title(f"h = {h:.5f} ({label})", fontsize=9)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Explicit Euler: h_crit = {h_crit:.5f}", fontsize=12, y=1.04)
    fig.tight_layout()
    plt.show()


def show_van_der_pol_stiff(mu: float = 1000.0, T: float = 3000.0):
    def vdp(t, y):
        return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])

    def vdp_jac(t, y):
        return np.array([
            [0.0, 1.0],
            [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0]**2)],
        ])

    y0 = np.array([2.0, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # implicit Euler
    h_impl = T / 2000
    t0 = time.perf_counter()
    res_impl = implicit_euler(vdp, y0, (0.0, T), h_impl, jac=vdp_jac)
    time_impl = time.perf_counter() - t0

    axes[0].plot(res_impl.t, res_impl.y[:, 0], "#9b59b6", lw=1.0)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"Implicit Euler (h={h_impl:.1f}, {len(res_impl.t)} steps, {time_impl*1000:.0f} ms)")
    axes[0].grid(True, alpha=0.3)

    # explicit Euler (will likely fail or need huge number of steps)
    h_expl = 1.0 / mu  # ~ stability limit
    n_steps_expl = int(T / h_expl)
    axes[1].text(0.5, 0.6, f"Explicit Euler would need\nh ≈ {h_expl:.1e}\n≈ {n_steps_expl:.0e} steps",
                transform=axes[1].transAxes, fontsize=14, ha="center", color="#e74c3c")
    axes[1].text(0.5, 0.3, f"vs Implicit Euler:\n{len(res_impl.t)} steps",
                transform=axes[1].transAxes, fontsize=14, ha="center", color="#9b59b6")
    axes[1].set_title(f"Van der Pol oscillator (μ = {mu})")
    axes[1].axis("off")

    fig.suptitle("Stiff ODE: Van der Pol oscillator", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.show()


def show_stiff_interactive():
    @interact(
        log_k=FloatSlider(min=1, max=6, step=0.5, value=4, description="log₁₀(k_stiff)"),
        log_h=FloatSlider(min=-4, max=-1, step=0.25, value=-2, description="log₁₀(h)"),
    )
    def _update(log_k, log_h):
        show_stiff_system_comparison(k_stiff=10**log_k, k_soft=1.0, T=10.0, h=10**log_h)
