"""Integration tests for autograder worker process_job."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from shared.schemas import Job
from autograder.worker import process_job


@pytest.fixture
def sample_job():
    return Job(
        chat_id=111,
        week_id="01",
        files={"beads.py": "def solve(): return 1"},
        user_id=222,
        first_name="Test",
        username="test",
    )


def test_process_job_mocked_docker(sample_job, tmp_path, monkeypatch):
    """process_job runs Docker, parses output, stores grades, sends Telegram."""
    monkeypatch.setenv("GRADES_DB_PATH", str(tmp_path / "grades.db"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    import autograder.grades.schema as mod
    mod._schema_initialized.clear()

    stdout = """
test_beads.py::test_x PASSED
test_broom_racing.py::test_y FAILED
test_so101_ik.py::test_z PASSED
"""
    with patch("autograder.worker.docker_run") as m_docker:
        m_docker.return_value = (1, stdout, "")
        with patch("autograder.worker._send_telegram") as m_send:
            process_job(sample_job)

    m_docker.assert_called_once_with("01", sample_job.files)
    m_send.assert_called_once()
    call_args = m_send.call_args
    assert call_args[0][0] == 111  # chat_id
    assert "fail" in call_args[0][1].lower() or "❌" in call_args[0][1]


def test_process_job_docker_timeout(sample_job, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    import subprocess

    with patch("autograder.worker.docker_run") as m_docker:
        m_docker.side_effect = subprocess.TimeoutExpired("docker", 10)
        with patch("autograder.worker._send_telegram") as m_send:
            process_job(sample_job)

    m_send.assert_called_once()
    assert "timeout" in m_send.call_args[0][1].lower() or "timed out" in m_send.call_args[0][1].lower()
