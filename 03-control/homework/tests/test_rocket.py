"""Open tests for the A* Ship (rocket TVC) problem."""
from __future__ import annotations

import numpy as np

from lib.rocket import RocketEnv, RocketEnvConfig, WindConfig, make_ascent_scenario, make_landing_scenario
from solutions.rocket import RocketController


def _run_episode(controller: RocketController, config: RocketEnvConfig) -> tuple[bool, dict]:
    env = RocketEnv(config)
    if hasattr(controller, "reset"):
        controller.reset()
    obs, info = env.reset()

    max_steps = int(np.ceil(env.scenario.time_limit / env.config.dt))
    for _ in range(max_steps):
        try:
            action = np.asarray(controller(obs), dtype=float)
        except NotImplementedError as exc:
            raise AssertionError(
                "Implement RocketController.__call__(obs) to return "
                "[thrust, gimbal_angle]."
            ) from exc
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["status"] == "success" and reward == 1.0, info

    return False, info


def test_rocket_controller_succeeds_on_ascent_and_landing():
    controller = RocketController()
    cases = [
        ("ascent-default", RocketEnvConfig(scenario=make_ascent_scenario(), wind=WindConfig(seed=98765))),
        (
            "ascent-altitude-angle",
            RocketEnvConfig(
                scenario=make_ascent_scenario(z_target=120.0, target_theta=0.10),
                wind=WindConfig(seed=24680),
            ),
        ),
        ("landing-default", RocketEnvConfig(scenario=make_landing_scenario(), wind=WindConfig(seed=98765))),
        (
            "landing-shifted-pad",
            RocketEnvConfig(
                scenario=make_landing_scenario(target_x=14.0),
                wind=WindConfig(seed=24680),
            ),
        ),
    ]
    results = []

    for label, config in cases:
        success, info = _run_episode(controller, config)
        results.append((label, success, info["status"]))

    success_rate = sum(success for _, success, _ in results) / len(results)
    assert success_rate == 1.0, (
        "RocketController must complete fixed and varied rocket scenarios. "
        f"success_rate={success_rate:.2f}, results={results}"
    )
