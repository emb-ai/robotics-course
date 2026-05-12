"""Tests for batch homework discovery."""

from pathlib import Path

import yaml

from autograder.batch.discovery import discover_homeworks, get_homework


def test_discovers_local_homeworks_including_unregistered_week(repo_root):
    result = discover_homeworks(repo_root)

    assert {"01", "02", "03"}.issubset(result.homeworks)
    assert result.homeworks["03"].topic_slug == "03-control"
    assert result.homeworks["01"].compose_file.endswith(
        "01-intro-and-kinematics/homework/container/docker_compose.yaml"
    )


def test_default_test_dependencies_use_problem_id_solution_file(repo_root):
    spec = get_homework(repo_root, "01")

    assert spec.test_dependencies["test_beads.py"] == ["beads.py"]
    assert spec.test_dependencies["test_broom_racing.py"] == ["broom_racing.py"]


def test_week02_joints_declares_multi_file_dependencies(repo_root):
    spec = get_homework(repo_root, "02")

    assert spec.test_dependencies["test_joints.py"] == [
        "constraints.py",
        "constraints_manager.py",
        "ode_solvers.py",
    ]


def test_explicit_test_dependencies_override_defaults(tmp_path):
    homework_dir = tmp_path / "02-dynamics" / "homework"
    container_dir = homework_dir / "container"
    container_dir.mkdir(parents=True)
    (container_dir / "docker_compose.yaml").write_text("services: {}\n")
    (homework_dir / "autograder.yaml").write_text(
        yaml.safe_dump(
            {
                "solution_files": ["constraints.py", "constraints_manager.py", "ode_solvers.py"],
                "problem_ids": {"test_joints.py": "joints"},
                "points": {"joints": 5},
                "test_dependencies": {
                    "test_joints.py": [
                        "constraints.py",
                        "constraints_manager.py",
                        "ode_solvers.py",
                    ]
                },
            }
        )
    )

    spec = get_homework(tmp_path, "02")

    assert spec.test_dependencies["test_joints.py"] == [
        "constraints.py",
        "constraints_manager.py",
        "ode_solvers.py",
    ]


def test_discovery_reports_warnings_without_crashing(tmp_path):
    homework_dir = tmp_path / "04-broken" / "homework"
    homework_dir.mkdir(parents=True)
    (homework_dir / "autograder.yaml").write_text(
        yaml.safe_dump({"solution_files": ["answer.py"], "problem_ids": {"test_answer.py": "answer"}})
    )

    result = discover_homeworks(tmp_path)

    assert "04" in result.homeworks
    assert any("docker_compose.yaml" in warning for warning in result.warnings)
