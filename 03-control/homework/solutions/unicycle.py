"""Student solution stub for the unicycle torque-control problem."""
from __future__ import annotations

import numpy as np


class UnicycleController:
    """Stateful controller interface for the unicycle task.

    The observation is a flat array:
    ``[target_x, target_y, qpos(10), qvel(9)]``.

    Expanded observation order:
    ``[target_x, target_y,
    wheel_x, wheel_y, wheel_z,
    wheel_qw, wheel_qx, wheel_qy, wheel_qz,
    wheel_rotation_angle, pelvis_y_angle, pelvis_x_angle,
    wheel_vx, wheel_vy, wheel_vz,
    wheel_omega_x, wheel_omega_y, wheel_omega_z,
    wheel_angular_velocity, pelvis_y_angular_velocity,
    pelvis_x_angular_velocity]``.

    Return raw motor torques in this order:
    ``[tau_pelvis_y, tau_pelvis_x, tau_wheel]``.
    """

    def reset(self) -> None:
        """Clear any per-episode controller state if needed."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
