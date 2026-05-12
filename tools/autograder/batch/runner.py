"""CLI and orchestration for reports-only batch grading."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autograder.pytest_parser import parse_metrics
from autograder.diagnostics import run_diagnostics_for_student

from .artifacts import BatchArtifactStore
from .discovery import get_homework
from .docker_runner import DockerRunResult, prebuild_homework_image, run_student_tests
from .junit_parser import parse_junit_results
from .models import BatchConfig, BatchState, ProblemRunResult, SubmittedStudent, to_jsonable
from .selector import TestSelection, select_tests_for_student
from .submissions import scan_submissions


def run_batch(
    homework_id: str,
    submissions_root: str | Path,
    output_root: str | Path,
    run_id: str,
    max_workers: int = 2,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one local grading batch and write artifacts."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    spec = get_homework(root, homework_id)
    started_at = _now()
    config = BatchConfig(
        run_id=run_id,
        homework_id=homework_id,
        submissions_root=str(Path(submissions_root)),
        output_root=str(Path(output_root)),
        max_workers=max_workers,
        created_at=started_at,
    )
    store = BatchArtifactStore(output_root, run_id)
    store.write_config(config)
    store.update_state(
        BatchState(
            status="building",
            counts={"total": 0, "completed": 0, "failed": 0, "scan_error": 0},
            started_at=started_at,
        )
    )

    students = scan_submissions(submissions_root, spec.solution_files)
    counts = {"total": len(students), "completed": 0, "failed": 0, "scan_error": 0}
    store.update_state(BatchState(status="building", counts=counts, started_at=started_at))
    prebuild_homework_image(spec, repo_root=root)

    results: list[dict[str, Any]] = []
    active: set[str] = set()
    workers = max(1, int(max_workers or 1))
    store.update_state(BatchState(status="running", counts=counts, started_at=started_at, active_jobs=[]))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for student in students:
            for name, content in student.files.items():
                store.write_submitted_file(student.student_id, name, content)
            if student.errors:
                result = _scan_error_result(spec, student)
                _attach_diagnostics(root, spec, result, store)
                store.write_student_result(student.student_id, result)
                results.append(result)
                counts["scan_error"] += 1
                counts["completed"] += 1
                continue

            selection = select_tests_for_student(spec, student)
            result_dir = store.student_dir(student.student_id)
            future = executor.submit(_run_one_student, root, spec, student, selection, result_dir, store)
            futures[future] = student.student_id
            active.add(student.student_id)
            store.update_state(
                BatchState(
                    status="running",
                    counts=counts,
                    started_at=started_at,
                    active_jobs=sorted(active),
                )
            )

        for future in as_completed(futures):
            student_id = futures[future]
            active.discard(student_id)
            try:
                result = future.result()
            except Exception as exc:
                result = _internal_error_result(spec, student_id, str(exc))
            results.append(result)
            if result["status"] != "passed":
                counts["failed"] += 1
            counts["completed"] += 1
            store.update_state(
                BatchState(
                    status="running",
                    counts=counts,
                    started_at=started_at,
                    active_jobs=sorted(active),
                )
            )

    batch_result = {
        "homework_id": spec.id,
        "problem_ids": list(spec.problem_ids.values()),
        "points": spec.points,
        "students": sorted(results, key=lambda item: item["student_id"]),
    }
    store.write_results(batch_result)
    store.write_summary_csv(batch_result)
    store.write_index_html(batch_result)
    store.update_state(
        BatchState(
            status="done",
            counts=counts,
            started_at=started_at,
            finished_at=_now(),
            active_jobs=[],
        )
    )
    return to_jsonable(batch_result)


def _run_one_student(
    repo_root: Path,
    spec,
    student: SubmittedStudent,
    selection: TestSelection,
    result_dir: Path,
    store: BatchArtifactStore,
) -> dict[str, Any]:
    if not selection.selected_tests:
        docker_result = DockerRunResult(exit_code=0, stdout="", stderr="", elapsed_sec=0.0, pytest_xml="")
    else:
        docker_raw = run_student_tests(
            spec,
            student.student_id,
            student.files,
            selection.selected_tests,
            result_dir,
            repo_root=repo_root,
        )
        docker_result = _coerce_docker_result(docker_raw, result_dir)

    store.write_student_output(
        student.student_id,
        stdout=docker_result.stdout,
        stderr=docker_result.stderr,
        pytest_xml=docker_result.pytest_xml,
    )
    problems = parse_junit_results(
        docker_result.pytest_xml,
        spec.problem_ids,
        spec.points,
        missing_results=selection.missing_results,
    )
    problems = _fill_unreported_selected_problems(spec, selection, problems, docker_result)
    metrics = parse_metrics(docker_result.stdout, docker_result.stderr)
    result = _student_result(
        spec=spec,
        student=student,
        selection=selection,
        problems=problems,
        metrics=metrics,
        exit_code=docker_result.exit_code,
        elapsed_sec=docker_result.elapsed_sec,
    )
    _attach_diagnostics(repo_root, spec, result, store)
    store.write_student_result(student.student_id, result)
    return result


def _coerce_docker_result(raw: Any, result_dir: Path) -> DockerRunResult:
    if isinstance(raw, DockerRunResult):
        return raw
    if isinstance(raw, dict):
        xml_path = result_dir / "pytest.xml"
        pytest_xml = str(raw.get("pytest_xml", ""))
        if not pytest_xml and xml_path.exists():
            pytest_xml = xml_path.read_text(encoding="utf-8", errors="replace")
        return DockerRunResult(
            exit_code=int(raw.get("exit_code", 1)),
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            elapsed_sec=float(raw.get("elapsed_sec", 0.0)),
            pytest_xml=pytest_xml,
        )
    raise TypeError(f"unexpected docker result: {raw!r}")


def _fill_unreported_selected_problems(
    spec,
    selection: TestSelection,
    problems: list[ProblemRunResult],
    docker_result: DockerRunResult,
) -> list[ProblemRunResult]:
    seen = {problem.test_file for problem in problems}
    selected_files = {Path(path).name for path in selection.selected_tests}
    for test_file, problem_id in spec.problem_ids.items():
        if test_file not in selected_files or test_file in seen:
            continue
        status = "passed" if docker_result.exit_code == 0 else "error"
        max_points = float(spec.points.get(problem_id, 0))
        problems.append(
            ProblemRunResult(
                problem_id=problem_id,
                test_file=test_file,
                status=status,
                points_possible=max_points,
                points_earned=max_points if status == "passed" else 0,
                message="No JUnit testcase was reported for this selected test.",
            )
        )
    return problems


def _student_result(
    spec,
    student: SubmittedStudent,
    selection: TestSelection,
    problems: list[ProblemRunResult],
    metrics: dict[str, float],
    exit_code: int | None,
    elapsed_sec: float,
) -> dict[str, Any]:
    problem_map = {
        problem.problem_id: {
            "test_file": problem.test_file,
            "status": problem.status,
            "points": problem.points_earned,
            "max_points": problem.points_possible,
            "message": problem.message,
        }
        for problem in problems
    }
    status = "passed" if problem_map and all(item["status"] == "passed" for item in problem_map.values()) else "failed"
    return {
        "student_id": student.student_id,
        "status": status,
        "submitted_files": sorted(student.files),
        "ignored_files": student.ignored_files,
        "missing_files": selection.missing_files,
        "selected_tests": selection.selected_tests,
        "problems": problem_map,
        "metrics": metrics,
        "exit_code": exit_code,
        "elapsed_sec": elapsed_sec,
        "artifacts": [],
    }


def _attach_diagnostics(
    repo_root: Path,
    spec,
    result: dict[str, Any],
    store: BatchArtifactStore,
) -> None:
    if result.get("status") == "passed":
        result.setdefault("diagnostics", [])
        return
    try:
        diagnostics = run_diagnostics_for_student(
            repo_root=repo_root,
            spec=spec,
            student_result=result,
            artifact_store=store,
        )
    except Exception as exc:
        diagnostics = [
            {
                "plugin_id": "diagnostics",
                "problem_id": "",
                "status": "error",
                "summary": "Diagnostics failed.",
                "metrics": {},
                "artifacts": [],
                "error": str(exc),
            }
        ]
    result["diagnostics"] = diagnostics
    artifacts = result.setdefault("artifacts", [])
    for diagnostic in diagnostics:
        artifacts.extend(diagnostic.get("artifacts") or [])


def _scan_error_result(spec, student: SubmittedStudent) -> dict[str, Any]:
    problems = {
        problem_id: {
            "test_file": test_file,
            "status": "error",
            "points": 0,
            "max_points": float(spec.points.get(problem_id, 0)),
            "message": "; ".join(student.errors),
        }
        for test_file, problem_id in spec.problem_ids.items()
    }
    return {
        "student_id": student.student_id,
        "status": "scan_error",
        "submitted_files": [],
        "ignored_files": student.ignored_files,
        "missing_files": [],
        "selected_tests": [],
        "problems": problems,
        "metrics": {},
        "exit_code": None,
        "elapsed_sec": 0.0,
        "artifacts": [],
    }


def _internal_error_result(spec, student_id: str, message: str) -> dict[str, Any]:
    problems = {
        problem_id: {
            "test_file": test_file,
            "status": "error",
            "points": 0,
            "max_points": float(spec.points.get(problem_id, 0)),
            "message": message,
        }
        for test_file, problem_id in spec.problem_ids.items()
    }
    return {
        "student_id": student_id,
        "status": "error",
        "submitted_files": [],
        "ignored_files": [],
        "missing_files": [],
        "selected_tests": [],
        "problems": problems,
        "metrics": {},
        "exit_code": None,
        "elapsed_sec": 0.0,
        "artifacts": [],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reports-only local batch grading.")
    parser.add_argument("--homework", required=True)
    parser.add_argument("--submissions-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        run_batch(
            args.homework,
            args.submissions_root,
            args.output_root,
            args.run_id,
            max_workers=args.max_workers,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Batch setup failed: {exc}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
