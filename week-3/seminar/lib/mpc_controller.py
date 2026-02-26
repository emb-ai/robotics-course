from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.linalg
from scipy.optimize import minimize

from .controllers import Controller


class MPCController(Controller):
    def __init__(
        self,
        env,
        N: int = 30,
        Q_diag: tuple[float, ...] = (10, 100, 10, 100),
        R_val: float = 0.005,
        Qf_scale: float = 20.0,
        f_max: float = 10.0,
        dt: float = 0.02,
        target_state: NDArray | None = None,
    ):
        self.N = N
        self.Q = np.diag(Q_diag)
        self.Qf = Qf_scale * self.Q
        self.R_val = R_val
        self.f_max = f_max
        self.dt = dt
        self.target_state = target_state if target_state is not None else np.zeros(4)

        A_c, B_c = env.linearize()
        n = A_c.shape[0]
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A_c * dt
        M[:n, n:] = B_c * dt
        eM = scipy.linalg.expm(M)
        self.A_d: NDArray = eM[:n, :n]
        self.B_d: NDArray = eM[:n, n:]

    def __call__(self, t: float, state: NDArray) -> float:
        N = self.N
        u0 = np.zeros(N)
        bounds = [(-self.f_max, self.f_max)] * N
        x_ref = self.target_state

        def cost(u_seq: NDArray) -> float:
            x = state.copy()
            total = 0.0
            for k in range(N):
                dx = x - x_ref
                total += float(dx @ self.Q @ dx + self.R_val * u_seq[k] ** 2)
                x = self.A_d @ x + self.B_d.ravel() * u_seq[k]
            dx = x - x_ref
            total += float(dx @ self.Qf @ dx)
            return total

        def grad(u_seq: NDArray) -> NDArray:
            xs = [state.copy()]
            for k in range(N):
                xs.append(self.A_d @ xs[-1] + self.B_d.ravel() * u_seq[k])

            lam = 2.0 * self.Qf @ (xs[N] - x_ref)
            g = np.zeros(N)
            for k in range(N - 1, -1, -1):
                g[k] = float((self.B_d.T @ lam).item()) + 2.0 * self.R_val * u_seq[k]
                lam = 2.0 * self.Q @ (xs[k] - x_ref) + self.A_d.T @ lam
            return g

        result = minimize(
            cost, u0, jac=grad, method="SLSQP", bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        return float(result.x[0])


def _state_error(x: NDArray, x_ref: NDArray) -> NDArray:
    err = x - x_ref
    theta_err = ((x[1] - x_ref[1]) + np.pi) % (2 * np.pi) - np.pi
    err[1] = theta_err
    return err


class NonlinearMPCController(Controller):
    def __init__(
        self,
        env,
        N: int = 50,
        Q_diag: tuple[float, ...] = (10, 100, 10, 100),
        R_val: float = 0.005,
        Qf_scale: float = 100.0,
        f_max: float = 10.0,
        target_state: NDArray | None = None,
    ):
        self.env = env
        self.N = N
        self.Q = np.diag(Q_diag)
        self.Qf = Qf_scale * self.Q
        self.R_val = R_val
        self.f_max = f_max
        self.target_state = target_state if target_state is not None else np.zeros(4)
        self._u_prev: NDArray | None = None

    def reset(self):
        self._u_prev = None

    def __call__(self, t: float, state: NDArray) -> float:
        N = self.N
        x_ref = self.target_state

        if self._u_prev is not None and len(self._u_prev) == N:
            u0 = np.concatenate([self._u_prev[1:], [self._u_prev[-1]]])
        else:
            u0 = np.zeros(N)

        bounds = [(-self.f_max, self.f_max)] * N

        def cost(u_seq: NDArray) -> float:
            x = state.copy()
            total = 0.0
            for k in range(N):
                err = _state_error(x, x_ref)
                total += float(err @ self.Q @ err + self.R_val * u_seq[k] ** 2)
                x = self.env._rk4_step(x, float(u_seq[k]))
            err = _state_error(x, x_ref)
            total += float(err @ self.Qf @ err)
            return total

        result = minimize(
            cost, u0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-6},
        )
        self._u_prev = result.x.copy()
        return float(result.x[0])
