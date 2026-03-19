"""Tests for shared/week_config.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from shared import week_config


def test_load_weeks_yaml_returns_dict():
    data = week_config.load_weeks_yaml()
    assert isinstance(data, dict)
    assert "01" in data or 1 in data


def test_get_week_config_valid():
    cfg = week_config.get_week_config("01")
    assert "compose_file" in cfg
    assert "topic_slug" in cfg
    assert cfg["topic_slug"] == "01-intro-and-kinematics"


def test_get_week_config_int_key():
    """YAML may parse '01' as int 1."""
    cfg = week_config.get_week_config("01")
    assert cfg is not None


def test_get_week_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown week_id"):
        week_config.get_week_config("99")


def test_get_topic_slug():
    slug = week_config.get_topic_slug("01")
    assert slug == "01-intro-and-kinematics"


def test_get_solution_files():
    files = week_config.get_solution_files("01")
    assert "beads.py" in files
    assert "broom_racing.py" in files
    assert "so101_ik.py" in files


def test_get_problem_ids():
    ids = week_config.get_problem_ids("01")
    assert ids.get("test_beads.py") == "beads"
    assert ids.get("test_broom_racing.py") == "broom_racing"


def test_get_points():
    pts = week_config.get_points("01")
    assert pts.get("beads") == 10
    assert pts.get("broom_racing") == 15


def test_get_limits():
    limits = week_config.get_limits("01")
    assert limits["timeout_sec"] == 120
    assert limits["memory_mb"] == 512
    assert limits["cpus"] == 1


def test_list_weeks():
    weeks = week_config.list_weeks()
    assert isinstance(weeks, list)
    assert "01" in weeks


def test_get_repo_root_from_env(monkeypatch):
    monkeypatch.setenv("AI_ROBOTICS_REPO_ROOT", "/tmp/custom_root")
    root = week_config.get_repo_root()
    assert str(root) == "/tmp/custom_root"


def test_get_repo_root_inferred():
    root = week_config.get_repo_root()
    assert root.exists()
    assert (root / "tools").exists()
    assert (root / "tools" / "config" / "weeks.yaml").exists()
