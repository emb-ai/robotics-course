from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, NamedTuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from ipywidgets import interact, FloatSlider, IntSlider
import scipy.linalg


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

class CartPoleParams(NamedTuple):
    m_cart: float = 1.0
    m_pole: float = 0.1
    l: float = 0.5        # half-length of pole
    g: float = 9.81
    f_max: float = 10.0   # force limit (for MPC constraints)


DEFAULT_PARAMS = CartPoleParams()


# ---------------------------------------------------------------------------
# State: [x, theta, x_dot, theta_dot]
#   x       – cart position
#   theta   – pole angle from upright (0 = up)
#   x_dot   – cart velocity
#   theta_dot – pole angular velocity
# ---------------------------------------------------------------------------

def cartpole_dynamics(state: NDArray, u: float, p: CartPoleParams = DEFAULT_PARAMS) -> NDArray:
    x, theta, x_dot, theta_dot = state
    mc, mp, l, g = p.m_cart, p.m_pole, p.l, p.g
    total_mass = mc + mp
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = total_mass - mp * cos_t ** 2

    theta_ddot = (
        (total_mass * g * sin_t
         - cos_t * (u + mp * l * theta_dot ** 2 * sin_t))
        / (l * denom)
    )
    x_ddot = (
        (u + mp * l * (theta_dot ** 2 * sin_t - theta_ddot * cos_t))
        / total_mass
    )
    return np.array([x_dot, theta_dot, x_ddot, theta_ddot])


def cartpole_rhs(p: CartPoleParams = DEFAULT_PARAMS):
    def f(t: float, state: NDArray, u_fn: Callable | None = None) -> NDArray:
        u = 0.0 if u_fn is None else u_fn(t, state)
        return cartpole_dynamics(state, u, p)
    return f


# ---------------------------------------------------------------------------
# RK4 integration
# ---------------------------------------------------------------------------

def rk4_step(f: Callable, t: float, y: NDArray, h: float, *args) -> NDArray:
    k1 = f(t, y, *args)
    k2 = f(t + h / 2, y + h / 2 * k1, *args)
    k3 = f(t + h / 2, y + h / 2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_cartpole(
    state0: NDArray,
    controller: Callable[[float, NDArray], float],
    T: float = 5.0,
    dt: float = 0.02,
    p: CartPoleParams = DEFAULT_PARAMS,
) -> tuple[NDArray, NDArray, NDArray]:
    n_steps = int(T / dt)
    ts = np.linspace(0, T, n_steps + 1)
    states = np.zeros((n_steps + 1, 4))
    controls = np.zeros(n_steps + 1)
    states[0] = state0
    controls[0] = controller(0.0, state0)

    for i in range(n_steps):
        u = controller(ts[i], states[i])
        u = np.clip(u, -p.f_max, p.f_max)
        controls[i] = u

        def f_closed(t, s, *_args):
            return cartpole_dynamics(s, u, p)

        states[i + 1] = rk4_step(f_closed, ts[i], states[i], dt)

    controls[-1] = controller(ts[-1], states[-1])
    return ts, states, controls


# ---------------------------------------------------------------------------
# Linearization about upright equilibrium (theta=0, theta_dot=0)
# ---------------------------------------------------------------------------

def linearize_cartpole(p: CartPoleParams = DEFAULT_PARAMS) -> tuple[NDArray, NDArray]:
    mc, mp, l, g = p.m_cart, p.m_pole, p.l, p.g
    total_mass = mc + mp

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, -mp * g / mc, 0, 0],
        [0, total_mass * g / (mc * l), 0, 0],
    ])
    B = np.array([
        [0],
        [0],
        [1 / mc],
        [-1 / (mc * l)],
    ])
    return A, B


# ---------------------------------------------------------------------------
# Energy computations
# ---------------------------------------------------------------------------

def cartpole_energy(state: NDArray, p: CartPoleParams = DEFAULT_PARAMS) -> float:
    _, theta, x_dot, theta_dot = state
    mc, mp, l, g = p.m_cart, p.m_pole, p.l, p.g
    KE_cart = 0.5 * mc * x_dot ** 2
    v_pole_x = x_dot + l * theta_dot * np.cos(theta)
    v_pole_y = -l * theta_dot * np.sin(theta)
    KE_pole = 0.5 * mp * (v_pole_x ** 2 + v_pole_y ** 2)
    PE_pole = mp * g * l * np.cos(theta)
    return KE_cart + KE_pole + PE_pole


def upright_energy(p: CartPoleParams = DEFAULT_PARAMS) -> float:
    return p.m_pole * p.g * p.l


# ---------------------------------------------------------------------------
# Visualization / Animation
# ---------------------------------------------------------------------------

def draw_cartpole(ax: plt.Axes, state: NDArray, p: CartPoleParams = DEFAULT_PARAMS,
                  x_lim: float = 3.0):
    ax.clear()
    x, theta = state[0], state[1]
    cart_w, cart_h = 0.4, 0.2
    pole_len = 2 * p.l

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)

    cart = Rectangle((x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                      facecolor="steelblue", edgecolor="black", zorder=3)
    ax.add_patch(cart)

    tip_x = x + pole_len * np.sin(theta)
    tip_y = pole_len * np.cos(theta)
    ax.plot([x, tip_x], [0, tip_y], "o-", color="firebrick", lw=3, markersize=8, zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("z")


def animate_cartpole(ts: NDArray, states: NDArray, controls: NDArray | None = None,
                     p: CartPoleParams = DEFAULT_PARAMS,
                     x_lim: float = 3.0, skip: int = 2) -> HTML:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_cart, ax_states, ax_ctrl = axes

    idx = list(range(0, len(ts), skip))
    if idx[-1] != len(ts) - 1:
        idx.append(len(ts) - 1)

    ax_states.set_xlim(ts[0], ts[-1])
    y_margin = 0.5
    ax_states.set_ylim(states[:, 1].min() - y_margin, states[:, 1].max() + y_margin)
    ax_states.set_xlabel("t [s]")
    ax_states.set_ylabel("state")
    ax_states.grid(True, alpha=0.3)
    line_x, = ax_states.plot([], [], label="x")
    line_theta, = ax_states.plot([], [], label=r"$\theta$")
    ax_states.legend(loc="upper right", fontsize=8)

    if controls is not None:
        ax_ctrl.set_xlim(ts[0], ts[-1])
        ax_ctrl.set_ylim(controls.min() - 1, controls.max() + 1)
        ax_ctrl.set_xlabel("t [s]")
        ax_ctrl.set_ylabel("u [N]")
        ax_ctrl.grid(True, alpha=0.3)
        line_u, = ax_ctrl.plot([], [], color="green", label="u")
        ax_ctrl.legend(loc="upper right", fontsize=8)
    else:
        ax_ctrl.set_visible(False)

    def update(frame):
        i = idx[frame]
        draw_cartpole(ax_cart, states[i], p, x_lim)
        ax_cart.set_title(f"t = {ts[i]:.2f} s")
        line_x.set_data(ts[:i + 1], states[:i + 1, 0])
        line_theta.set_data(ts[:i + 1], states[:i + 1, 1])
        if controls is not None:
            line_u.set_data(ts[:i + 1], controls[:i + 1])
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), interval=40, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# ---------------------------------------------------------------------------
# Quick interactive demo: manual slider control
# ---------------------------------------------------------------------------

def show_cartpole_demo(p: CartPoleParams = DEFAULT_PARAMS):
    fig, ax = plt.subplots(figsize=(6, 3))

    def _update(x=0.0, theta_deg=10.0):
        draw_cartpole(ax, np.array([x, np.radians(theta_deg), 0, 0]), p)
        fig.canvas.draw_idle()

    interact(_update,
             x=FloatSlider(min=-2, max=2, step=0.1, value=0, description="x"),
             theta_deg=FloatSlider(min=-180, max=180, step=1, value=10, description="θ [deg]"))
    plt.show()


# ---------------------------------------------------------------------------
# LQR solver (convenience wrapper)
# ---------------------------------------------------------------------------

def solve_lqr(A: NDArray, B: NDArray, Q: NDArray, R: NDArray) -> tuple[NDArray, NDArray]:
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P
