"""Tests for dependency-aware batch test selection."""

from autograder.batch.models import HomeworkSpec, SubmittedStudent
from autograder.batch.selector import select_tests_for_student


def _spec() -> HomeworkSpec:
    return HomeworkSpec(
        id="02",
        topic_slug="02-dynamics",
        homework_dir="/repo/02-dynamics/homework",
        compose_file="/repo/02-dynamics/homework/container/docker_compose.yaml",
        solution_files=["kin_energy.py", "constraints.py", "constraints_manager.py", "ode_solvers.py"],
        problem_ids={"test_kin_energy.py": "kin_energy", "test_joints.py": "joints"},
        points={"kin_energy": 5, "joints": 5},
        metrics={},
        limits={"timeout_sec": 120, "memory_mb": 512, "cpus": 1, "network": "none"},
        test_dependencies={
            "test_kin_energy.py": ["kin_energy.py"],
            "test_joints.py": ["constraints.py", "constraints_manager.py", "ode_solvers.py"],
        },
    )


def test_selects_tests_when_all_dependencies_are_present():
    student = SubmittedStudent(
        student_id="Alice",
        source_dir="/tmp/Alice",
        files={
            "kin_energy.py": "",
            "constraints.py": "",
            "constraints_manager.py": "",
            "ode_solvers.py": "",
        },
        ignored_files=[],
        errors=[],
    )

    selection = select_tests_for_student(_spec(), student)

    assert selection.selected_tests == ["tests/test_kin_energy.py", "tests/test_joints.py"]
    assert selection.missing_results == []


def test_marks_missing_dependencies_without_selecting_test():
    student = SubmittedStudent(
        student_id="Alice",
        source_dir="/tmp/Alice",
        files={"constraints.py": "", "ode_solvers.py": ""},
        ignored_files=[],
        errors=[],
    )

    selection = select_tests_for_student(_spec(), student)

    assert selection.selected_tests == []
    assert {result.problem_id for result in selection.missing_results} == {"kin_energy", "joints"}
    joints = next(result for result in selection.missing_results if result.problem_id == "joints")
    assert joints.status == "missing"
    assert "constraints_manager.py" in joints.message
