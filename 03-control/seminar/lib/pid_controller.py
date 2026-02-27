from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .controllers import Controller


class PIDController(Controller):
    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        dt: float = 0.02,
        output_limits: tuple[float, float] = (-10.0, 10.0),
        setpoint: float = 0.0,
        derivative_filter_coeff: float = 0.1,
        x_setpoint: float = 0.0,
        Kp_x: float = 0.0,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits
        self.setpoint = setpoint
        self.alpha = derivative_filter_coeff
        self.x_setpoint = x_setpoint
        self.Kp_x = Kp_x

        self._integral: float = 0.0
        self._prev_error: float | None = None
        self._filtered_derivative: float = 0.0

    def __call__(self, t: float, state: NDArray) -> float:
        theta = state[1]
        error = theta - self.setpoint

        self._integral += error * self.dt
        integral_limit = self.output_limits[1] / max(abs(self.Ki), 1e-8)
        self._integral = np.clip(self._integral, -integral_limit, integral_limit)

        if self._prev_error is None:
            raw_derivative = 0.0
        else:
            raw_derivative = (error - self._prev_error) / self.dt

        self._filtered_derivative = (
            self.alpha * raw_derivative + (1 - self.alpha) * self._filtered_derivative
        )
        self._prev_error = error

        x_error = state[0] - self.x_setpoint
        output = (
            self.Kp * error
            + self.Ki * self._integral
            + self.Kd * self._filtered_derivative
            + self.Kp_x * x_error
        )
        return float(np.clip(output, *self.output_limits))

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0
