from __future__ import annotations

import numpy as np

JOINT_LIMIT_RADIUS = np.pi / 2


def optimal_worm_config(link_lengths: np.ndarray) -> np.ndarray:
    """Return joint angles (2*(N-1),) in rad that minimize AABB diagonal.
    Satisfies: circular joint limits, no non-consecutive self-collision."""
    raise NotImplementedError
