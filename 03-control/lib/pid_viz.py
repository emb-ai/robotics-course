from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, FloatLogSlider

from .cartpole_sim import (
    DEFAULT_PARAMS, CartPoleParams,
    simulate_cartpole, linearize_cartpole,
)

# ---------------------------------------------------------------------------
# Mass-spring-damper plant for PID demos: m*x_ddot + c*x_dot + k*x = u
# ---------------------------------------------------------------------------
_MSD_M, _MSD_C, _MSD_K = 1.0, 0.5, 1.0


def _simulate_msd_pid(
    Kp: float, Ki: float, Kd: float,
    T: float = 10.0, dt: float = 0.005, ref: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    n = int(T / dt)
    ts = np.linspace(0, T, n)
    y = np.zeros(n)
    u = np.zeros(n)
    err = np.zeros(n)

    x1, x2 = 0.0, 0.0  # position, velocity
    integral = 0.0
    prev_err = ref

    for i in range(n):
        e = ref - x1
        err[i] = e
        integral += e * dt
        de = (e - prev_err) / dt if i > 0 else 0.0
        prev_err = e

        ctrl = Kp * e + Ki * integral + Kd * de
        u[i] = ctrl
        y[i] = x1

        x1_ddot = (ctrl - _MSD_C * x2 - _MSD_K * x1) / _MSD_M
        x2 += x1_ddot * dt
        x1 += x2 * dt

    return ts, y, u, err


def _step_metrics(ts: NDArray, y: NDArray, ref: float = 1.0) -> dict:
    tol = 0.02 * ref
    rise_time = settling_time = np.nan
    overshoot = 0.0

    above_10 = np.where(y >= 0.1 * ref)[0]
    above_90 = np.where(y >= 0.9 * ref)[0]
    if len(above_10) and len(above_90):
        rise_time = ts[above_90[0]] - ts[above_10[0]]

    peak = np.max(y)
    if peak > ref:
        overshoot = (peak - ref) / ref * 100

    for i in range(len(ts) - 1, -1, -1):
        if abs(y[i] - ref) > tol:
            settling_time = ts[min(i + 1, len(ts) - 1)]
            break

    return {"rise_time": rise_time, "overshoot": overshoot, "settling_time": settling_time}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_pid_step_response_interactive():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    def _update(Kp: float = 4.0, Ki: float = 1.0, Kd: float = 1.5):
        ts, y, u, err = _simulate_msd_pid(Kp, Ki, Kd)
        m = _step_metrics(ts, y)

        ax1.clear()
        ax1.plot(ts, y, "b", lw=2, label="Output y(t)")
        ax1.axhline(1.0, color="gray", ls="--", lw=1, label="Reference")
        ax1.fill_between(ts, 0.98, 1.02, alpha=0.1, color="green", label="±2 % band")
        if not np.isnan(m["rise_time"]):
            ax1.annotate(f"Rise: {m['rise_time']:.2f}s", xy=(m["rise_time"], 0.9),
                         fontsize=9, color="purple", fontweight="bold")
        if m["overshoot"] > 0:
            ax1.annotate(f"OS: {m['overshoot']:.1f}%", xy=(ts[np.argmax(y)], np.max(y)),
                         fontsize=9, color="red", fontweight="bold")
        if not np.isnan(m["settling_time"]):
            ax1.axvline(m["settling_time"], color="green", ls=":", alpha=0.7)
            ax1.annotate(f"Settle: {m['settling_time']:.2f}s", xy=(m["settling_time"], 0.5),
                         fontsize=9, color="green", fontweight="bold")
        ax1.set_ylabel("y(t)")
        ax1.set_title(f"PID Step Response  (Kp={Kp}, Ki={Ki}, Kd={Kd})")
        ax1.legend(fontsize=8, loc="lower right")
        ax1.grid(True, alpha=0.3)

        ax2.clear()
        ax2.plot(ts, u, "g", lw=1.5)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("u(t)")
        ax2.set_title("Control signal")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw_idle()

    interact(
        _update,
        Kp=FloatSlider(value=4.0, min=0.1, max=15.0, step=0.1, description="Kp"),
        Ki=FloatSlider(value=1.0, min=0.0, max=5.0, step=0.1, description="Ki"),
        Kd=FloatSlider(value=1.5, min=0.0, max=8.0, step=0.1, description="Kd"),
    )
    plt.show()


def show_pid_step_response(Kp: float = 4.0, Ki: float = 1.0, Kd: float = 1.5):
    ts, y, u, err = _simulate_msd_pid(Kp, Ki, Kd)
    m = _step_metrics(ts, y)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(ts, y, "b", lw=2, label="Output y(t)")
    ax1.axhline(1.0, color="gray", ls="--", lw=1, label="Reference")
    ax1.fill_between(ts, 0.98, 1.02, alpha=0.1, color="green", label="±2 % band")
    if not np.isnan(m["rise_time"]):
        ax1.annotate(f"Rise: {m['rise_time']:.2f}s", xy=(m["rise_time"], 0.9),
                     fontsize=9, color="purple", fontweight="bold")
    if m["overshoot"] > 0:
        ax1.annotate(f"OS: {m['overshoot']:.1f}%", xy=(ts[np.argmax(y)], np.max(y)),
                     fontsize=9, color="red", fontweight="bold")
    if not np.isnan(m["settling_time"]):
        ax1.axvline(m["settling_time"], color="green", ls=":", alpha=0.7)
        ax1.annotate(f"Settle: {m['settling_time']:.2f}s", xy=(m["settling_time"], 0.5),
                     fontsize=9, color="green", fontweight="bold")
    ax1.set_ylabel("y(t)")
    ax1.set_title(f"PID Step Response  (Kp={Kp}, Ki={Ki}, Kd={Kd})")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, u, "g", lw=1.5)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("u(t)")
    ax2.set_title("Control signal")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_pid_gain_effects():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Kp sweep
    ax = axes[0, 0]
    for Kp in [0.5, 1, 2, 4, 8]:
        ts, y, _, _ = _simulate_msd_pid(Kp, 0, 0)
        ax.plot(ts, y, label=f"Kp={Kp}")
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.set_title("Effect of Kp (Ki=0, Kd=0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("y(t)")

    # Ki sweep
    ax = axes[0, 1]
    for Ki in [0, 0.2, 0.5, 1.0, 2.0]:
        ts, y, _, _ = _simulate_msd_pid(2.0, Ki, 0)
        ax.plot(ts, y, label=f"Ki={Ki}")
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.set_title("Effect of Ki (Kp=2, Kd=0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Kd sweep
    ax = axes[1, 0]
    for Kd in [0, 0.3, 1.0, 2.0, 4.0]:
        ts, y, _, _ = _simulate_msd_pid(4.0, 0, Kd)
        ax.plot(ts, y, label=f"Kd={Kd}")
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.set_title("Effect of Kd (Kp=4, Ki=0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y(t)")

    # Well-tuned
    ax = axes[1, 1]
    ts, y, _, _ = _simulate_msd_pid(4.0, 1.0, 1.5)
    ax.plot(ts, y, "b", lw=2)
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.fill_between(ts, 0.98, 1.02, alpha=0.1, color="green")
    ax.set_title("Well-tuned PID (Kp=4, Ki=1, Kd=1.5)")
    ax.set_xlabel("t [s]")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PID Gain Effects on Mass-Spring-Damper Step Response",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


def show_pid_cartpole_interactive(p: CartPoleParams = DEFAULT_PARAMS):
    fig, (ax_states, ax_ctrl) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    def _update(Kp=30.0, Ki=0.0, Kd=10.0, theta0_deg=10.0):
        theta0 = np.radians(theta0_deg)
        state0 = np.array([0.0, theta0, 0.0, 0.0])

        integral = [0.0]
        prev_err = [theta0]

        def controller(t, state):
            e = -state[1]
            integral[0] += e * 0.02
            integral[0] = np.clip(integral[0], -5.0, 5.0)
            de = (e - prev_err[0]) / 0.02
            prev_err[0] = e
            return Kp * e + Ki * integral[0] + Kd * de

        ts, states, controls = simulate_cartpole(state0, controller, T=5.0, dt=0.02, p=p)

        ax_states.clear()
        ax_states.plot(ts, states[:, 0], label="x [m]")
        ax_states.plot(ts, np.degrees(states[:, 1]), label="θ [deg]")
        ax_states.axhline(0, color="gray", ls="--", lw=0.5)
        ax_states.set_ylabel("State")
        ax_states.set_title(f"PID on CartPole (Kp={Kp:.1f}, Ki={Ki:.1f}, Kd={Kd:.1f})")
        ax_states.legend(fontsize=8)
        ax_states.grid(True, alpha=0.3)

        ax_ctrl.clear()
        ax_ctrl.plot(ts, controls, "g", lw=1.5)
        ax_ctrl.axhline(p.f_max, color="r", ls=":", alpha=0.5, label=f"±{p.f_max} N")
        ax_ctrl.axhline(-p.f_max, color="r", ls=":", alpha=0.5)
        ax_ctrl.set_xlabel("t [s]")
        ax_ctrl.set_ylabel("u [N]")
        ax_ctrl.legend(fontsize=8)
        ax_ctrl.grid(True, alpha=0.3)
        fig.canvas.draw_idle()

    interact(
        _update,
        Kp=FloatLogSlider(value=30, base=10, min=-1, max=2, step=0.05, description="Kp"),
        Ki=FloatSlider(value=0, min=0, max=20, step=0.5, description="Ki"),
        Kd=FloatLogSlider(value=10, base=10, min=-1, max=1.7, step=0.05, description="Kd"),
        theta0_deg=FloatSlider(value=10, min=1, max=45, step=1, description="θ₀ [deg]"),
    )
    plt.show()


def show_ziegler_nichols_demo():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    Kp_values = [0.5, 1.0, 2.0, 4.0, 6.5]
    ax = axes[0]
    for Kp in Kp_values:
        ts, y, _, _ = _simulate_msd_pid(Kp, 0, 0, T=15.0)
        ax.plot(ts, y, label=f"Kp={Kp}")
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.set_title("Step 1: Increase Kp until sustained oscillation")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y(t)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    Ku = 6.5
    ts_osc, y_osc, _, _ = _simulate_msd_pid(Ku, 0, 0, T=20.0)
    peaks_idx = []
    for i in range(1, len(y_osc) - 1):
        if y_osc[i] > y_osc[i - 1] and y_osc[i] > y_osc[i + 1] and ts_osc[i] > 2.0:
            peaks_idx.append(i)
    Tu = np.mean(np.diff(ts_osc[peaks_idx])) if len(peaks_idx) > 2 else 2.0

    ax = axes[1]
    ax.plot(ts_osc, y_osc, "b", lw=1.5)
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    for pi in peaks_idx[:5]:
        ax.plot(ts_osc[pi], y_osc[pi], "ro", markersize=6)
    ax.set_title(f"Ku ≈ {Ku:.1f}, Tu ≈ {Tu:.2f} s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y(t)")
    ax.grid(True, alpha=0.3)

    Kp_zn = 0.6 * Ku
    Ki_zn = 2 * Kp_zn / Tu
    Kd_zn = Kp_zn * Tu / 8

    ax = axes[2]
    ts_zn, y_zn, _, _ = _simulate_msd_pid(Kp_zn, Ki_zn, Kd_zn, T=15.0)
    ax.plot(ts_zn, y_zn, "b", lw=2, label="ZN-tuned PID")
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.fill_between(ts_zn, 0.98, 1.02, alpha=0.1, color="green")
    ax.set_title(f"ZN PID: Kp={Kp_zn:.2f}, Ki={Ki_zn:.2f}, Kd={Kd_zn:.2f}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y(t)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Ziegler-Nichols Tuning Method", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

    print(f"Ziegler-Nichols result: Ku={Ku:.2f}, Tu={Tu:.2f}s")
    print(f"  PID gains: Kp={Kp_zn:.3f}, Ki={Ki_zn:.3f}, Kd={Kd_zn:.3f}")
