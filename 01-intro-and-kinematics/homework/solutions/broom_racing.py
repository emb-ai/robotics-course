from typing import Callable

import numpy as np

from lib.broom_racing import Configuration, XYZConfiguration


def gate_pass(
    start: Configuration,
    goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    """
    Fly from start to goal through the Quidditch gate (contest 1).

    Parameters
    ----------
    start : Configuration
        Initial pose (x, y, z, θ, φ). Speed v=1; θ heading, φ pitch.
    goal : Configuration
        Gate pose to pass through (position and orientation).

    Returns
    -------
    callable
        curve(s) for s ∈ [0, 1], returning Configuration. Must satisfy discretized EOM
        (ẋ=v cos θ cos φ, ẏ=v sin θ cos φ, ż=v sin φ, θ̇=u₁/cos φ, φ̇=u₂), pitch φ ∈ [φ_min, φ_max],
        and curvature κ = √(u₁² + u₂²) ≤ κ_max. Path length ≤ reference implementation.
    """
    raise NotImplementedError


def catch_snitch(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> Callable[[np.ndarray], Configuration]:
    """
    Reach the Snitch at goal_xyz (contest 2); only position matters at goal.

    Parameters
    ----------
    start : Configuration
        Initial pose (x, y, z, θ, φ).
    goal_xyz : XYZConfiguration
        Target position (x, y, z) to reach.

    Returns
    -------
    callable
        curve(s) for s ∈ [0, 1], returning Configuration. Same EOM and constraint
        requirements as gate_pass; path length ≤ reference.
    """
    raise NotImplementedError


def catch_ball_and_gate(
    start: Configuration,
    intermediate_goal_xyz: XYZConfiguration,
    final_goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    """
    Catch the ball at intermediate_goal_xyz then fly through the gate at final_goal (contest 3).

    Parameters
    ----------
    start : Configuration
        Initial pose.
    intermediate_goal_xyz : XYZConfiguration
        Ball position (x, y, z) to visit first.
    final_goal : Configuration
        Gate pose to pass through after catching the ball.

    Returns
    -------
    callable
        curve(s) for s ∈ [0, 1], returning Configuration. Curve visits ball then gate;
        same EOM and constraint requirements; path length ≤ reference.
    """
    raise NotImplementedError
