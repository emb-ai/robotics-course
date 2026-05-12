"""Tests for the reports-only batch Docker runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from autograder.batch.docker_runner import run_student_tests
from autograder.batch.models import HomeworkSpec


def _spec(tmp_path: Path) -> HomeworkSpec:
    homework_dir = tmp_path / "01-intro-and-kinematics" / "homework"
    compose_file = homework_dir / "container" / "docker_compose.yaml"
    compose_file.parent.mkdir(parents=True)
    compose_file.write_text("services: {}\n")
    return HomeworkSpec(
        id="01",
        topic_slug="01-intro-and-kinematics",
        homework_dir=str(homework_dir),
        compose_file=str(compose_file),
        solution_files=["beads.py"],
        problem_ids={"test_beads.py": "beads"},
        points={"beads": 10},
        metrics={},
        limits={"timeout_sec": 120, "memory_mb": 512, "cpus": 1, "network": "none"},
        test_dependencies={"test_beads.py": ["beads.py"]},
    )


def test_batch_docker_runner_uses_unique_project_result_mount_and_junit(tmp_path):
    result_dir = tmp_path / "results"
    with patch("autograder.batch.docker_runner.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        result = run_student_tests(
            _spec(tmp_path),
            student_id="Alice",
            files={"beads.py": "answer = 42\n"},
            selected_tests=["tests/test_beads.py"],
            result_dir=result_dir,
            repo_root=tmp_path,
        )

    cmd = run_mock.call_args[0][0]
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert "-p" in cmd
    assert any(str(arg).startswith("batch-01-alice-") for arg in cmd)
    assert f"{result_dir.resolve()}:/results" in cmd
    assert "--junitxml=/results/pytest.xml" in cmd
    assert "GRADING_STUDENT_SUBMISSION=1" in cmd
    assert not any("TELEGRAM" in key or "REDIS" in key for key in run_mock.call_args[1]["env"])


def test_prebuild_homework_image_runs_compose_build(tmp_path):
    from autograder.batch.docker_runner import prebuild_homework_image

    with patch("autograder.batch.docker_runner.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0, stdout="built", stderr="")

        prebuild_homework_image(_spec(tmp_path), repo_root=tmp_path)

    cmd = run_mock.call_args[0][0]
    assert cmd[-2:] == ["build", "homework-tests"]
