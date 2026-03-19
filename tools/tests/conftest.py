"""Pytest fixtures for tools tests."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tools_root() -> Path:
    """Repo tools/ directory (parent of tests/)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root(tools_root) -> Path:
    """Repo root (parent of tools/)."""
    return tools_root.parent


@pytest.fixture
def temp_dir():
    """Temporary directory, cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def monkeypatch_env(monkeypatch):
    """Helper to patch env vars, restoring after test."""

    def _set(key: str, value: str | None):
        monkeypatch.setitem(os.environ, key, value)

    return _set


@pytest.fixture
def mock_weeks_yaml(temp_dir):
    """Create a minimal weeks.yaml for testing week_config."""
    config_dir = temp_dir / "config"
    config_dir.mkdir()
    yaml_path = config_dir / "weeks.yaml"
    yaml_path.write_text('''
"01":
  compose_file: 01-intro-and-kinematics/homework/container/docker_compose.yaml
  topic_slug: 01-intro-and-kinematics
  solution_files:
    - beads.py
    - broom_racing.py
    - so101_ik.py
  problem_ids:
    test_beads.py: beads
    test_broom_racing.py: broom_racing
    test_so101_ik.py: so101_ik
''')
    return yaml_path
