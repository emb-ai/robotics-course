from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Controller:
    def __call__(self, t: float, state: NDArray) -> float:
        raise NotImplementedError

    def reset(self):
        pass


class ZeroController(Controller):
    def __call__(self, t: float, state: NDArray) -> float:
        return 0.0
