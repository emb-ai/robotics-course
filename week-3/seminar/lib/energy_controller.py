from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .controllers import Controller
from .lqr_controller import LQRController


class EnergySwingUpController(Controller):
    def __init__(
        self,
        env,
        k_energy: float = 20.0,
        k_x: float = 0.5,
        k_xd: float = 0.5,
        lqr_Q_diag: tuple[float, ...] = (1, 10, 1, 10),
        lqr_R_val: float = 1.0,
        switch_theta: float = 0.3,
        switch_theta_dot: float = 2.0,
        f_max: float = 10.0,
        target_x: float = 0.0,
    ):
        self.env = env
        self.k_energy = k_energy
        self.k_x = k_x
        self.k_xd = k_xd
        self.switch_theta = switch_theta
        self.switch_theta_dot = switch_theta_dot
        self.f_max = f_max
        self.target_x = target_x
        self.E_target = env.upright_energy()

        target_state = np.array([target_x, 0.0, 0.0, 0.0])
        self.lqr = LQRController(
            env_or_AB=env,
            Q_diag=lqr_Q_diag,
            R_val=lqr_R_val,
            f_max=f_max,
            target_state=target_state,
        )

    def _use_lqr(self, state: NDArray) -> bool:
        theta = state[1]
        theta_dot = state[3]
        return abs(theta) < self.switch_theta and abs(theta_dot) < self.switch_theta_dot

    def __call__(self, t: float, state: NDArray) -> float:
        if self._use_lqr(state):
            return self.lqr(t, state)

        E = self.env.energy(state)
        theta = state[1]
        theta_dot = state[3]
        x, x_dot = state[0], state[2]
        u_energy = self.k_energy * (E - self.E_target) * theta_dot * np.cos(theta)
        u_cart = -self.k_x * (x - self.target_x) - self.k_xd * x_dot
        return float(np.clip(u_energy + u_cart, -self.f_max, self.f_max))

    def reset(self):
        self.lqr.reset()
