from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.linalg

from .controllers import Controller


class LQRController(Controller):
    def __init__(
        self,
        env_or_AB: tuple[NDArray, NDArray] | None = None,
        Q_diag: tuple[float, ...] = (1, 10, 1, 10),
        R_val: float = 1.0,
        f_max: float = 10.0,
        target_state: NDArray | None = None,
    ):
        if env_or_AB is None:
            from .cartpole_env import CartPoleEnv
            A, B = CartPoleEnv().linearize()
        elif isinstance(env_or_AB, tuple):
            A, B = env_or_AB
        else:
            A, B = env_or_AB.linearize()

        Q = np.diag(Q_diag)
        R = np.array([[R_val]])
        self.f_max = f_max
        self.target_state = target_state if target_state is not None else np.zeros(4)

        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K: NDArray = np.linalg.solve(R, B.T @ P)

    def __call__(self, t: float, state: NDArray) -> float:
        u = float((-self.K @ (state - self.target_state)).item())
        return float(np.clip(u, -self.f_max, self.f_max))
