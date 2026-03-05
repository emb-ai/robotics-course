from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VelocityMotionModel:
    """Probabilistic velocity-based motion model for a diff-drive/unicycle robot.

    Noise parameters alpha_1..4 follow Probabilistic Robotics (Thrun et al.) convention:
        alpha_1, alpha_2 : noise in angular velocity (omega)
        alpha_3, alpha_4 : noise in linear velocity (v)
    """
    alpha_1: float = 0.05   # v noise from v
    alpha_2: float = 0.01   # v noise from omega
    alpha_3: float = 0.01   # omega noise from v
    alpha_4: float = 0.05   # omega noise from omega

    def sample(
        self,
        state: np.ndarray,
        v: float,
        omega: float,
        dt: float,
        n: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample n next states from p(x_t | x_{t-1}, u_t).

        Returns shape (n, 3) if n > 1 else (3,).
        """
        rng = rng or np.random.default_rng()
        std_v = np.sqrt(self.alpha_1 * v**2 + self.alpha_2 * omega**2)
        std_w = np.sqrt(self.alpha_3 * v**2 + self.alpha_4 * omega**2)
        v_hat = v + rng.normal(0, std_v, size=n)
        omega_hat = omega + rng.normal(0, std_w, size=n)

        x, y, th = state
        if n == 1:
            v_hat, omega_hat = v_hat[0], omega_hat[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(np.abs(omega_hat) < 1e-9, np.inf, v_hat / omega_hat)
            x_new = np.where(
                np.abs(omega_hat) < 1e-9,
                x + v_hat * np.cos(th) * dt,
                x - r * np.sin(th) + r * np.sin(th + omega_hat * dt),
            )
            y_new = np.where(
                np.abs(omega_hat) < 1e-9,
                y + v_hat * np.sin(th) * dt,
                y + r * np.cos(th) - r * np.cos(th + omega_hat * dt),
            )
            th_new = _wrap_angle(th + omega_hat * dt)

        if n == 1:
            return np.array([x_new, y_new, th_new])
        return np.stack([x_new, y_new, th_new], axis=1)

    def log_prob(
        self,
        x_prev: np.ndarray,
        x_next: np.ndarray,
        v: float,
        omega: float,
        dt: float,
    ) -> float:
        """Approximate log p(x_next | x_prev, v, omega) via inverse kinematics."""
        x1, y1, th1 = x_prev
        x2, y2, th2 = x_next
        dx, dy = x2 - x1, y2 - y1
        mu = 0.5 * (x1 * dy - y1 * dx - x2 * dy + y2 * dx) / (
            dy * np.cos(th1) - dx * np.sin(th1) + 1e-12
        )
        x_c = (x1 + x2) / 2 + mu * dy
        y_c = (y1 + y2) / 2 - mu * dx
        r = np.sqrt((x1 - x_c)**2 + (y1 - y_c)**2)
        delta_th = _wrap_angle(th2 - th1)
        v_hat = r * delta_th / dt if abs(delta_th) > 1e-9 else np.hypot(dx, dy) / dt
        omega_hat = delta_th / dt

        var_v = self.alpha_1 * v**2 + self.alpha_2 * omega**2
        var_w = self.alpha_3 * v**2 + self.alpha_4 * omega**2

        lp_v = -0.5 * (v_hat - v)**2 / (var_v + 1e-12) - 0.5 * np.log(2 * np.pi * var_v + 1e-12)
        lp_w = -0.5 * (omega_hat - omega)**2 / (var_w + 1e-12) - 0.5 * np.log(2 * np.pi * var_w + 1e-12)
        return lp_v + lp_w


@dataclass
class OdometryMotionModel:
    """Probabilistic odometry-based motion model.

    Decomposes motion as rot1 -> trans -> rot2.
    Noise parameters alpha_1..4 (Thrun et al. convention):
        alpha_1: rot noise from rot
        alpha_2: rot noise from trans
        alpha_3: trans noise from trans
        alpha_4: trans noise from rot
    """
    alpha_1: float = 0.05
    alpha_2: float = 0.01
    alpha_3: float = 0.01
    alpha_4: float = 0.05

    @staticmethod
    def odometry_to_motion(
        x_odom_prev: np.ndarray,
        x_odom_curr: np.ndarray,
    ) -> tuple[float, float, float]:
        """Extract (delta_rot1, delta_trans, delta_rot2) from two odometry poses."""
        dx = x_odom_curr[0] - x_odom_prev[0]
        dy = x_odom_curr[1] - x_odom_prev[1]
        delta_trans = np.hypot(dx, dy)
        delta_rot1 = _wrap_angle(np.arctan2(dy, dx) - x_odom_prev[2])
        delta_rot2 = _wrap_angle(x_odom_curr[2] - x_odom_prev[2] - delta_rot1)
        return delta_rot1, delta_trans, delta_rot2

    def sample(
        self,
        state: np.ndarray,
        delta_rot1: float,
        delta_trans: float,
        delta_rot2: float,
        n: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample n next states from p(x_t | x_{t-1}, u_t).

        Returns shape (n, 3) if n > 1 else (3,).
        """
        rng = rng or np.random.default_rng()
        std_r1 = np.sqrt(self.alpha_1 * delta_rot1**2 + self.alpha_2 * delta_trans**2)
        std_t  = np.sqrt(self.alpha_3 * delta_trans**2 + self.alpha_4 * (delta_rot1**2 + delta_rot2**2))
        std_r2 = np.sqrt(self.alpha_1 * delta_rot2**2 + self.alpha_2 * delta_trans**2)

        dr1_hat = delta_rot1  - rng.normal(0, std_r1, size=n)
        dt_hat  = delta_trans - rng.normal(0, std_t,  size=n)
        dr2_hat = delta_rot2  - rng.normal(0, std_r2, size=n)

        x, y, th = state
        th_mid = th + dr1_hat
        x_new  = x + dt_hat * np.cos(th_mid)
        y_new  = y + dt_hat * np.sin(th_mid)
        th_new = _wrap_angle(th_mid + dr2_hat)

        if n == 1:
            return np.array([x_new[0], y_new[0], th_new[0]])
        return np.stack([x_new, y_new, th_new], axis=1)

    def log_prob(
        self,
        x_prev: np.ndarray,
        x_next: np.ndarray,
        delta_rot1: float,
        delta_trans: float,
        delta_rot2: float,
    ) -> float:
        dx = x_next[0] - x_prev[0]
        dy = x_next[1] - x_prev[1]
        dt_hat  = np.hypot(dx, dy)
        dr1_hat = _wrap_angle(np.arctan2(dy, dx) - x_prev[2])
        dr2_hat = _wrap_angle(x_next[2] - x_prev[2] - dr1_hat)

        var_r1 = self.alpha_1 * delta_rot1**2 + self.alpha_2 * delta_trans**2
        var_t  = self.alpha_3 * delta_trans**2 + self.alpha_4 * (delta_rot1**2 + delta_rot2**2)
        var_r2 = self.alpha_1 * delta_rot2**2 + self.alpha_2 * delta_trans**2

        def _lp(val, mean, var):
            return -0.5 * (val - mean)**2 / (var + 1e-12) - 0.5 * np.log(2 * np.pi * var + 1e-12)

        return (
            _lp(dr1_hat, delta_rot1, var_r1) +
            _lp(dt_hat,  delta_trans, var_t) +
            _lp(dr2_hat, delta_rot2, var_r2)
        )
