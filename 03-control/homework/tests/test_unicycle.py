"""Open tests for the student unicycle controller."""
from __future__ import annotations

import numpy as np

from lib.unicycle import FIXED_TARGET_CASES, UnicycleEnv, UnicycleEnvConfig
from solutions.unicycle import UnicycleController


def _run_episode(controller: UnicycleController, target_xy: tuple[float, float]) -> tuple[bool, dict]:
    env = UnicycleEnv(UnicycleEnvConfig(target_xy=target_xy))
    if hasattr(controller, "reset"):
        controller.reset()
    obs, info = env.reset(options={"target_xy": target_xy})

    max_steps = int(np.ceil(env.config.time_limit / env.policy_dt))
    for _ in range(max_steps):
        try:
            action = np.asarray(controller(obs), dtype=float)
        except NotImplementedError as exc:
            raise AssertionError(
                "Implement UnicycleController.__call__(obs) to return "
                "[tau_pelvis_y, tau_pelvis_x, tau_wheel]."
            ) from exc
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["status"] == "success" and reward == 1.0, info

    return False, info


def test_unicycle_controller_success_rate_is_one_on_fixed_resets():
    controller = UnicycleController()
    results = []

    for target_xy in FIXED_TARGET_CASES:
        success, info = _run_episode(controller, target_xy)
        results.append((target_xy, success, info["status"], info["distance_to_target"]))

    success_rate = sum(success for _, success, _, _ in results) / len(results)
    assert success_rate == 1.0, (
        "UnicycleController must reach every fixed target. "
        f"success_rate={success_rate:.2f}, results={results}"
    )
