from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Protocol
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class ControllerProto(Protocol):
    def __call__(self, t: float, state: NDArray) -> float: ...


class CartPoleEnv:
    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        l: float = 0.5,
        g: float = 9.81,
        dt: float = 0.02,
        f_max: float = 10.0,
        target_x: float = 0.0,
    ):
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.l = l
        self.g = g
        self.dt = dt
        self.f_max = f_max
        self.target_x = target_x
        self.state: NDArray = np.zeros(4)
        self.t: float = 0.0

    @property
    def target_state(self) -> NDArray:
        return np.array([self.target_x, 0.0, 0.0, 0.0])

    def reset(self, state: NDArray | None = None) -> NDArray:
        if state is not None:
            self.state = np.array(state, dtype=float)
        else:
            self.state = np.array([0.0, 0.05 * (2 * np.random.rand() - 1), 0.0, 0.0])
        self.t = 0.0
        return self.state.copy()

    def dynamics(self, state: NDArray, u: float) -> NDArray:
        x, theta, x_dot, theta_dot = state
        mc, mp, l, g = self.m_cart, self.m_pole, self.l, self.g
        total_mass = mc + mp
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        denom = l * (total_mass - mp * cos_t ** 2)

        theta_ddot = (
            total_mass * g * sin_t
            - cos_t * (u + mp * l * theta_dot ** 2 * sin_t)
        ) / denom
        x_ddot = (u + mp * l * (theta_dot ** 2 * sin_t - theta_ddot * cos_t)) / total_mass
        return np.array([x_dot, theta_dot, x_ddot, theta_ddot])

    def _rk4_step(self, state: NDArray, u: float) -> NDArray:
        h = self.dt
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + h / 2 * k1, u)
        k3 = self.dynamics(state + h / 2 * k2, u)
        k4 = self.dynamics(state + h * k3, u)
        return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, u: float) -> tuple[NDArray, float, bool, dict]:
        u = float(np.clip(u, -self.f_max, self.f_max))
        self.state = self._rk4_step(self.state, u)
        self.t += self.dt

        theta = self.state[1]
        x = self.state[0]
        reward = -(theta ** 2 + 0.1 * (x - self.target_x) ** 2 + 0.001 * u ** 2)
        done = abs(x) > 3.0
        return self.state.copy(), reward, done, {"t": self.t}

    def linearize(self) -> tuple[NDArray, NDArray]:
        mc, mp, l, g = self.m_cart, self.m_pole, self.l, self.g
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

    def energy(self, state: NDArray | None = None) -> float:
        if state is None:
            state = self.state
        _, theta, x_dot, theta_dot = state
        mc, mp, l, g = self.m_cart, self.m_pole, self.l, self.g
        KE_cart = 0.5 * mc * x_dot ** 2
        v_pole_x = x_dot + l * theta_dot * np.cos(theta)
        v_pole_y = -l * theta_dot * np.sin(theta)
        KE_pole = 0.5 * mp * (v_pole_x ** 2 + v_pole_y ** 2)
        PE_pole = mp * g * l * np.cos(theta)
        return float(KE_cart + KE_pole + PE_pole)

    def upright_energy(self) -> float:
        return self.m_pole * self.g * self.l

    def run_episode(
        self,
        controller: ControllerProto,
        T: float = 10.0,
        initial_state: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        state = self.reset(initial_state)
        n_steps = int(T / self.dt)
        ts = np.zeros(n_steps + 1)
        states = np.zeros((n_steps + 1, 4))
        controls = np.zeros(n_steps + 1)
        rewards = np.zeros(n_steps + 1)
        states[0] = state

        for i in range(n_steps):
            u = controller(self.t, self.state)
            controls[i] = np.clip(u, -self.f_max, self.f_max)
            ts[i] = self.t
            state, reward, done, _ = self.step(u)
            states[i + 1] = state
            rewards[i] = reward
            if done:
                ts[i + 1] = self.t
                controls[i + 1] = 0.0
                rewards[i + 1] = reward
                ts = ts[: i + 2]
                states = states[: i + 2]
                controls = controls[: i + 2]
                rewards = rewards[: i + 2]
                return ts, states, controls, rewards

        ts[n_steps] = self.t
        controls[n_steps] = controller(self.t, self.state)
        rewards[n_steps] = -(self.state[1] ** 2 + 0.1 * (self.state[0] - self.target_x) ** 2 + 0.001 * controls[n_steps] ** 2)
        return ts, states, controls, rewards

    def render(self, ax: plt.Axes | None = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))
        ax.clear()
        x, theta = self.state[0], self.state[1]
        cart_w, cart_h = 0.4, 0.2
        pole_len = 2 * self.l

        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        if self.target_x != 0.0:
            ax.axvline(self.target_x, color="red", ls="--", lw=1.5, alpha=0.6)
            ax.plot(self.target_x, pole_len, "r*", ms=12, zorder=5)

        cart = Rectangle(
            (x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
            facecolor="steelblue", edgecolor="black", zorder=3,
        )
        ax.add_patch(cart)

        tip_x = x + pole_len * np.sin(theta)
        tip_y = pole_len * np.cos(theta)
        ax.plot([x, tip_x], [0, tip_y], "o-", color="firebrick", lw=3, markersize=8, zorder=4)
        ax.set_xlabel("x")
        ax.set_ylabel("z")


def animate_episode(
    ts: NDArray,
    states: NDArray,
    controls: NDArray,
    env: CartPoleEnv,
    skip: int = 2,
) -> HTML:
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

    ax_ctrl.set_xlim(ts[0], ts[-1])
    ax_ctrl.set_ylim(controls.min() - 1, controls.max() + 1)
    ax_ctrl.set_xlabel("t [s]")
    ax_ctrl.set_ylabel("u [N]")
    ax_ctrl.grid(True, alpha=0.3)
    line_u, = ax_ctrl.plot([], [], color="green", label="u")
    ax_ctrl.legend(loc="upper right", fontsize=8)

    target_x = env.target_x

    def _draw_cart(ax: plt.Axes, state: NDArray):
        ax.clear()
        x, theta = state[0], state[1]
        cart_w, cart_h = 0.4, 0.2
        pole_len = 2 * env.l
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        if target_x != 0.0:
            ax.axvline(target_x, color="red", ls="--", lw=1.5, alpha=0.6, label=f"target x={target_x:.1f}")
            ax.plot(target_x, pole_len, "r*", ms=12, zorder=5)
        cart = Rectangle(
            (x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
            facecolor="steelblue", edgecolor="black", zorder=3,
        )
        ax.add_patch(cart)
        tip_x = x + pole_len * np.sin(theta)
        tip_y = pole_len * np.cos(theta)
        ax.plot([x, tip_x], [0, tip_y], "o-", color="firebrick", lw=3, markersize=8, zorder=4)
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    def update(frame: int):
        i = idx[frame]
        _draw_cart(ax_cart, states[i])
        ax_cart.set_title(f"t = {ts[i]:.2f} s")
        line_x.set_data(ts[: i + 1], states[: i + 1, 0])
        line_theta.set_data(ts[: i + 1], states[: i + 1, 1])
        line_u.set_data(ts[: i + 1], controls[: i + 1])
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), interval=40, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())
