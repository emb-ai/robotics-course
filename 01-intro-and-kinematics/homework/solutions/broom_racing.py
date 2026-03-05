from __future__ import annotations

from typing import Callable

import numpy as np

from lib.broom_types import Configuration, XYZConfiguration


def gate_pass(
    start: Configuration,
    goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    raise NotImplementedError


def catch_snitch(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> Callable[[np.ndarray], Configuration]:
    raise NotImplementedError


def catch_ball_and_gate(
    start: Configuration,
    intermediate_goal: XYZConfiguration,
    final_goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    raise NotImplementedError
