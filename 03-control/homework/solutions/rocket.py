"""Student solution stub for the rocket thrust-vector-control problem."""
from __future__ import annotations

import numpy as np

from lib.rocket import RocketParams

class RocketController:
    """Stateful controller interface for the rocket TVC task.

    The observation is a flat array:
    ``[target_x, target_z, target_theta, x, z, theta, vx, vz, omega]``.

    Return raw rocket controls in this order:
    ``[thrust, gimbal_angle]``.
    """

    def __init__(self, params: RocketParams | None = None):
        self.params = params if params is not None else RocketParams()

    def reset(self) -> None:
        """Clear any per-episode controller state if needed."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

