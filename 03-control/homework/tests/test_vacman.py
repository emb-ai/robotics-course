"""Open score tests for the Vacman control problem."""
from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lib.vacman import VacmanEnv, VacmanEnvConfig, load_vacman_cases
from solutions.vacman import VacmanController


PUBLIC_BASELINES = {
    "s_small": ("S", 177),
    "YSDA": ("YSDA", 3180),
    "empty": ("Empty", 1039),
}


def _public_cases() -> list[dict]:
    cases_by_id = {str(case["id"]): case for case in load_vacman_cases()}
    return [
        {
            "label": label,
            "baseline": baseline,
            "case": cases_by_id[case_id],
        }
        for case_id, (label, baseline) in PUBLIC_BASELINES.items()
    ]


try:
    HIDDEN_CASES = getattr(importlib.import_module("hidden_tests.test_vacman"), "HIDDEN_CASES", None)
except Exception:
    HIDDEN_CASES = None


def _all_cases() -> list[dict]:
    cases = _public_cases()
    if HIDDEN_CASES:
        cases.extend(HIDDEN_CASES)
    return cases


def _run_episode(controller: VacmanController, case_record: dict) -> tuple[bool, dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
        json.dump([case_record["case"]], f)
        f.flush()
        env = VacmanEnv(VacmanEnvConfig(cases_path=Path(f.name), case_index=0))
    if hasattr(controller, "reset"):
        controller.reset()
    obs, info = env.reset()
    max_steps = int(np.ceil(info["battery_capacity"] / env.config.dt)) + 1

    for _ in range(max_steps):
        try:
            action = np.asarray(controller(obs), dtype=float)
        except NotImplementedError as exc:
            raise AssertionError(
                "Implement VacmanController.__call__(obs) to return [v_left, v_right]."
            ) from exc
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["status"] == "success", info

    return False, info


@pytest.mark.parametrize("case_record", _all_cases(), ids=lambda c: c["label"])
def test_vacman_controller_beats_human_baseline(case_record: dict) -> None:
    controller = VacmanController()
    success, info = _run_episode(controller, case_record)
    label = case_record["label"]
    case_id = case_record["case"]["id"]
    baseline = int(case_record["baseline"])
    cleaned = int(info["cleaned"])
    details = (
        f"{label} ({case_id}): status={info['status']}, cleaned={cleaned}, "
        f"baseline={baseline}, t={info['t']:.2f}, "
        f"battery={info['battery_remaining']:.1f}/{info['battery_capacity']:.1f}, "
        f"base_distance={info['base_distance']:.3f}, cat_distance={info['cat_distance']:.3f}"
    )

    assert success, f"VacmanController must return to base successfully; {details}"
    assert cleaned > baseline, f"VacmanController must beat the human DUST baseline; {details}"
