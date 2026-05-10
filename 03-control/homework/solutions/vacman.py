"""Student solution stub for the Vacman control problem."""
from __future__ import annotations

import numpy as np


class VacmanController:
    """Stateful controller interface for the Vacman task.

    The observation is a dictionary with keys from
    ``lib.vacman.VACMAN_OBSERVATION_KEYS``. Return raw wheel speeds in this
    order: ``[v_left, v_right]``.
    """

    def reset(self) -> None:
        """Clear any per-episode controller state if needed."""

    def __call__(self, obs: dict) -> np.ndarray:
        raise NotImplementedError
