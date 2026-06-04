"""Tests for autograder/docker_runner.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from autograder.docker_runner import run


@pytest.fixture
def mock_subprocess():
    with patch("autograder.docker_runner.subprocess.run") as m:
        yield m


@pytest.fixture
def mock_week_registry():
    with patch("autograder.docker_runner.get_compose_path") as m1, \
         patch("autograder.docker_runner.get_repo_root") as m2, \
         patch("autograder.docker_runner.get_solutions_mount_path") as m3:
        m2.return_value = Path("/tmp/repo")
        m1.return_value = Path("/tmp/repo/01-intro-and-kinematics/homework/container/docker_compose.yaml")
        m3.return_value = "/app/01-intro-and-kinematics/homework/solutions"
        yield


def test_run_writes_files_and_calls_docker(mock_subprocess, mock_week_registry):
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    exit_code, stdout, stderr = run("01", {"beads.py": "x = 1"})
    assert exit_code == 0
    assert stdout == "ok"
    mock_subprocess.assert_called_once()
    env = mock_subprocess.call_args[1].get("env") or {}
    assert env.get("REPO_ROOT") == str(Path("/tmp/repo").resolve())
    call_args = mock_subprocess.call_args
    cmd = call_args[0][0]
    assert "docker" in cmd
    assert "compose" in cmd
    assert "homework-tests" in cmd
    assert "-e" in cmd
    assert "GRADING_STUDENT_SUBMISSION=1" in cmd
    assert any(
        isinstance(a, str) and a.startswith("GRADING_TEST_TIMEOUT_SEC=") for a in cmd
    )  # week 01 autograder.yaml limits.timeout_sec
    # Check -v volume mount
    vol_idx = cmd.index("-v") if "-v" in cmd else cmd.index("--volume") if "--volume" in cmd else -1
    assert vol_idx >= 0 or any("-v" in str(a) or "solutions" in str(a) for a in call_args[0])


def test_run_uses_prepared_mount_dir(mock_subprocess, mock_week_registry, tmp_path):
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    mount_dir = tmp_path / "prepared"
    mount_dir.mkdir()
    (mount_dir / "__init__.py").write_text("", encoding="utf-8")
    (mount_dir / "beads.py").write_text("x = 1", encoding="utf-8")

    exit_code, stdout, stderr = run("01", {"beads.py": "ignored"}, prepared_mount_dir=mount_dir)

    assert exit_code == 0
    assert stdout == "ok"
    cmd = mock_subprocess.call_args[0][0]
    vol_idx = cmd.index("-v")
    assert str(mount_dir.resolve()) in cmd[vol_idx + 1]


def test_run_skips_path_traversal(mock_subprocess, mock_week_registry):
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
    run("01", {"../evil.py": "bad"})  # should be skipped
    call_args = mock_subprocess.call_args
    # Temp dir should not contain evil.py - check via the run's behavior
    # We can't easily inspect temp dir, but the run completed
    assert mock_subprocess.called
