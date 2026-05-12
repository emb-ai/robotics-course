"""Dependency-aware visible test selection for batch grading."""

from __future__ import annotations

from dataclasses import dataclass

from .models import HomeworkSpec, ProblemRunResult, SubmittedStudent


@dataclass
class TestSelection:
    selected_tests: list[str]
    missing_results: list[ProblemRunResult]
    missing_files: list[str]


def select_tests_for_student(spec: HomeworkSpec, student: SubmittedStudent) -> TestSelection:
    """Select tests whose declared dependencies are all present."""

    submitted = set(student.files)
    selected: list[str] = []
    missing_results: list[ProblemRunResult] = []
    missing_files: list[str] = []

    for test_file, problem_id in spec.problem_ids.items():
        dependencies = spec.test_dependencies.get(test_file, [f"{problem_id}.py"])
        missing = [name for name in dependencies if name not in submitted]
        if missing:
            missing_files.extend(missing)
            missing_results.append(
                ProblemRunResult(
                    problem_id=problem_id,
                    test_file=test_file,
                    status="missing",
                    points_possible=float(spec.points.get(problem_id, 0)),
                    points_earned=0,
                    message=f"Missing dependencies: {', '.join(missing)}",
                )
            )
            continue
        selected.append(f"tests/{test_file}")

    return TestSelection(
        selected_tests=selected,
        missing_results=missing_results,
        missing_files=sorted(set(missing_files)),
    )
