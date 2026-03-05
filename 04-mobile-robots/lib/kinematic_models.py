from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class DubinsCar:
    """Dubins car: constant forward speed, bounded curvature.

    State: [x, y, theta]
    Control: kappa in [-kappa_max, kappa_max]  (signed curvature)
    """
    speed: float = 1.0
    kappa_max: float = 1.0

    @property
    def rho_min(self) -> float:
        return 1.0 / self.kappa_max

    def forward(self, state: np.ndarray, kappa: float, dt: float) -> np.ndarray:
        kappa = np.clip(kappa, -self.kappa_max, self.kappa_max)
        x, y, th = state
        v = self.speed
        omega = v * kappa
        if abs(omega) < 1e-9:
            x_new = x + v * np.cos(th) * dt
            y_new = y + v * np.sin(th) * dt
            th_new = th
        else:
            R = v / omega
            x_new = x + R * (np.sin(th + omega * dt) - np.sin(th))
            y_new = y - R * (np.cos(th + omega * dt) - np.cos(th))
            th_new = th + omega * dt
        return np.array([x_new, y_new, _wrap_angle(th_new)])


@dataclass
class DiffDrive:
    """Differential-drive (unicycle) kinematic model.

    State: [x, y, theta]
    Control: [v, omega]  (linear and angular velocity)
    """
    wheel_radius: float = 0.05   # metres
    wheel_base: float = 0.20     # metres (distance between wheels)

    def forward(
        self,
        state: np.ndarray,
        v: float,
        omega: float,
        dt: float,
    ) -> np.ndarray:
        x, y, th = state
        if abs(omega) < 1e-9:
            x_new = x + v * np.cos(th) * dt
            y_new = y + v * np.sin(th) * dt
            th_new = th
        else:
            r = v / omega
            x_new = x - r * np.sin(th) + r * np.sin(th + omega * dt)
            y_new = y + r * np.cos(th) - r * np.cos(th + omega * dt)
            th_new = th + omega * dt
        return np.array([x_new, y_new, _wrap_angle(th_new)])

    def wheels_to_twist(self, omega_r: float, omega_l: float) -> Tuple[float, float]:
        """Convert right/left wheel angular velocities to (v, omega)."""
        r, b = self.wheel_radius, self.wheel_base
        v = r * (omega_r + omega_l) / 2.0
        omega = r * (omega_r - omega_l) / b
        return v, omega

    def jacobian_state(self, state: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
        """G_t = df/dx  (3x3)."""
        _, _, th = state
        if abs(omega) < 1e-9:
            return np.array([
                [1, 0, -v * np.sin(th) * dt],
                [0, 1,  v * np.cos(th) * dt],
                [0, 0,  1],
            ])
        r = v / omega
        return np.array([
            [1, 0, -r * np.cos(th) + r * np.cos(th + omega * dt)],
            [0, 1, -r * np.sin(th) + r * np.sin(th + omega * dt)],
            [0, 0,  1],
        ])

    def jacobian_noise(self, state: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
        """V_t = df/dw  (3x2), where noise is in [v, omega]."""
        _, _, th = state
        if abs(omega) < 1e-9:
            return np.array([
                [np.cos(th) * dt, 0],
                [np.sin(th) * dt, 0],
                [0,               dt],
            ])
        r = v / omega
        drdv = 1.0 / omega
        drdw = -v / (omega ** 2)
        return np.array([
            [drdv * (-np.sin(th) + np.sin(th + omega * dt)),
             drdw * (-np.sin(th) + np.sin(th + omega * dt)) + r * np.cos(th + omega * dt) * dt],
            [drdv * (np.cos(th) - np.cos(th + omega * dt)),
             drdw * (np.cos(th) - np.cos(th + omega * dt)) + r * np.sin(th + omega * dt) * dt],
            [0, dt],
        ])


@dataclass
class AckermannModel:
    """Bicycle / Ackermann steering model (car-like).

    State: [x, y, theta]
    Control: [v, delta]  (speed and front steering angle)
    """
    wheelbase: float = 2.7       # metres (front-to-rear axle)
    max_steer: float = np.deg2rad(35.0)

    def forward(
        self,
        state: np.ndarray,
        v: float,
        delta: float,
        dt: float,
    ) -> np.ndarray:
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        x, y, th = state
        if abs(delta) < 1e-9:
            x_new = x + v * np.cos(th) * dt
            y_new = y + v * np.sin(th) * dt
            th_new = th
        else:
            R = self.wheelbase / np.tan(delta)
            beta = v / R * dt
            x_new = x + R * (np.sin(th + beta) - np.sin(th))
            y_new = y - R * (np.cos(th + beta) - np.cos(th))
            th_new = th + beta
        return np.array([x_new, y_new, _wrap_angle(th_new)])

    def turning_radius(self, delta: float) -> float:
        if abs(delta) < 1e-9:
            return np.inf
        return self.wheelbase / np.tan(delta)

    def jacobian_state(self, state: np.ndarray, v: float, delta: float, dt: float) -> np.ndarray:
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        _, _, th = state
        if abs(delta) < 1e-9:
            return np.array([
                [1, 0, -v * np.sin(th) * dt],
                [0, 1,  v * np.cos(th) * dt],
                [0, 0,  1],
            ])
        R = self.wheelbase / np.tan(delta)
        beta = v / R * dt
        return np.array([
            [1, 0, R * (np.cos(th + beta) - np.cos(th))],
            [0, 1, R * (np.sin(th + beta) - np.sin(th))],
            [0, 0, 1],
        ])


@dataclass
class OmniModel:
    """Omnidirectional (holonomic) kinematic model.

    State: [x, y, theta]
    Control: [vx, vy, omega]  in body frame
    Constraint matrix maps body-frame velocities to world-frame state derivatives.
    """

    def forward(
        self,
        state: np.ndarray,
        vx: float,
        vy: float,
        omega: float,
        dt: float,
    ) -> np.ndarray:
        x, y, th = state
        c, s = np.cos(th), np.sin(th)
        x_new = x + (c * vx - s * vy) * dt
        y_new = y + (s * vx + c * vy) * dt
        th_new = th + omega * dt
        return np.array([x_new, y_new, _wrap_angle(th_new)])

    def body_to_world(self, theta: float) -> np.ndarray:
        """3x3 rotation matrix mapping body-frame twist to world-frame derivative."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])


@dataclass
class MecanumModel:
    """Mecanum-wheel kinematic model (4 wheels).

    Wheel arrangement: front-left (FL), front-right (FR),
                       rear-left (RL), rear-right (RR).
    Control: [omega_FL, omega_FR, omega_RL, omega_RR]  (wheel angular velocities)
    Chassis velocity: [vx, vy, omega]  (body frame)
    """
    wheel_radius: float = 0.05
    lx: float = 0.15    # half-length (front/rear axle half-distance)
    ly: float = 0.10    # half-width  (left/right wheel half-distance)

    @property
    def _J(self) -> np.ndarray:
        """Jacobian J: chassis_vel = J @ wheel_omega  (shape 3x4)."""
        r = self.wheel_radius
        l = self.lx + self.ly
        return (r / 4) * np.array([
            [1,  1,  1,  1],
            [-1, 1,  1, -1],
            [-1/l, 1/l, -1/l, 1/l],
        ])

    @property
    def _J_inv(self) -> np.ndarray:
        """Pseudo-inverse: wheel_omega = J_inv @ chassis_vel  (shape 4x3)."""
        r = self.wheel_radius
        l = self.lx + self.ly
        return (1 / r) * np.array([
            [1, -1, -l],
            [1,  1,  l],
            [1,  1, -l],
            [1, -1,  l],
        ])

    def wheel_to_chassis(self, wheel_omega: np.ndarray) -> np.ndarray:
        """wheel_omega shape (4,) -> chassis_vel [vx, vy, omega]."""
        return self._J @ wheel_omega

    def chassis_to_wheel(self, chassis_vel: np.ndarray) -> np.ndarray:
        """chassis_vel [vx, vy, omega] -> wheel_omega (4,)."""
        return self._J_inv @ chassis_vel

    def forward(
        self,
        state: np.ndarray,
        wheel_omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        vx, vy, omega = self.wheel_to_chassis(wheel_omega)
        x, y, th = state
        c, s = np.cos(th), np.sin(th)
        x_new = x + (c * vx - s * vy) * dt
        y_new = y + (s * vx + c * vy) * dt
        th_new = th + omega * dt
        return np.array([x_new, y_new, _wrap_angle(th_new)])
