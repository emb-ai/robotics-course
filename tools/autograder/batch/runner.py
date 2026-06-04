"""CLI and orchestration for reports-only batch grading."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from autograder.pytest_parser import parse_metrics
from autograder.diagnostics import run_diagnostics_for_student
from autograder.feedback import generate_feedback_for_student

from .artifacts import BatchArtifactStore
from .discovery import get_homework
from .docker_runner import DockerRunResult, prebuild_homework_image, run_student_tests
from .junit_parser import parse_junit_results
from .models import BatchConfig, ProblemRunResult, SubmittedStudent, to_jsonable
from .selector import TestSelection, select_tests_for_student
from .submissions import scan_submissions


def run_batch(
    homework_id: str,
    submissions_root: str | Path,
    output_root: str | Path,
    run_id: str,
    max_workers: int = 2,
    repo_root: str | Path | None = None,
    enable_diagnostics: bool = True,
    enable_feedback: bool = True,
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
    progress = _ProgressWriter(
        store,
        started_at=started_at,
        counts={"total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "scan_error": 0},
    )
    progress.write("queued")

    students = scan_submissions(submissions_root, spec.solution_files)
    counts = {"total": len(students), "queued": len(students), "running": 0, "completed": 0, "failed": 0, "scan_error": 0}
    progress.counts = dict(counts)
    progress.student_phases = {student.student_id: "queued" for student in students}
    progress.write("building")
    prebuild_homework_image(spec, repo_root=root)

    results: list[dict[str, Any]] = []
    active: set[str] = set()
    workers = max(1, int(max_workers or 1))
    progress.write("running", active_jobs=[])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for student in students:
            for name, content in student.files.items():
                store.write_submitted_file(student.student_id, name, content)
            if student.errors:
                result = _scan_error_result(spec, student)
                result["student_path_id"] = store.student_path_id(student.student_id)
                if enable_diagnostics:
                    progress.set_student_phase(student.student_id, "diagnostics")
                    _attach_diagnostics(root, spec, result, store)
                else:
                    result["diagnostics"] = []
                if enable_feedback:
                    progress.set_student_phase(student.student_id, "feedback")
                    _attach_feedback(root, spec, result, store)
                else:
                    result["feedback"] = []
                store.write_student_result(student.student_id, result)
                results.append(result)
                progress.set_student_phase(student.student_id, "error")
                counts["queued"] -= 1
                counts["scan_error"] += 1
                counts["completed"] += 1
                progress.counts = dict(counts)
                progress.write("running", active_jobs=sorted(active))
                continue

            selection = select_tests_for_student(spec, student)
            result_dir = store.student_dir(student.student_id)
            future = executor.submit(
                _run_one_student,
                root,
                spec,
                student,
                selection,
                result_dir,
                store,
                progress,
                enable_diagnostics,
                enable_feedback,
            )
            futures[future] = student.student_id
            active.add(student.student_id)
            counts["queued"] -= 1
            counts["running"] += 1
            progress.counts = dict(counts)
            progress.write("running", active_jobs=sorted(active))

        for future in as_completed(futures):
            student_id = futures[future]
            active.discard(student_id)
            try:
                result = future.result()
            except Exception as exc:
                result = _internal_error_result(spec, student_id, str(exc))
                result["student_path_id"] = store.student_path_id(student_id)
                result["diagnostics"] = []
                result["feedback"] = []
                store.write_student_result(student_id, result)
                progress.set_student_phase(student_id, "error")
            results.append(result)
            if result["status"] != "passed":
                counts["failed"] += 1
            counts["running"] -= 1
            counts["completed"] += 1
            progress.counts = dict(counts)
            progress.write("running", active_jobs=sorted(active))

    batch_result = {
        "homework_id": spec.id,
        "problem_ids": list(spec.problem_ids.values()),
        "points": spec.points,
        "students": sorted(results, key=lambda item: item["student_id"]),
    }
    store.write_results(batch_result)
    store.write_summary_csv(batch_result)
    store.write_index_html(batch_result)
    progress.write("done", active_jobs=[], finished_at=_now())
    return to_jsonable(batch_result)


def _run_one_student(
    repo_root: Path,
    spec,
    student: SubmittedStudent,
    selection: TestSelection,
    result_dir: Path,
    store: BatchArtifactStore,
    progress: "_ProgressWriter",
    enable_diagnostics: bool,
    enable_feedback: bool,
) -> dict[str, Any]:
    progress.set_student_phase(student.student_id, "container")
    per_test_results: dict[str, DockerRunResult] = {}
    if not selection.selected_tests:
        docker_result = DockerRunResult(exit_code=0, stdout="", stderr="", elapsed_sec=0.0, pytest_xml="")
    else:
        docker_result, per_test_results = _run_selected_tests(
            repo_root,
            spec,
            student,
            selection,
            result_dir,
        )

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
    problems = _fill_unreported_selected_problems(spec, selection, problems, docker_result, per_test_results)
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
    result["student_path_id"] = store.student_path_id(student.student_id)
    if enable_diagnostics:
        progress.set_student_phase(student.student_id, "diagnostics")
        _attach_diagnostics(repo_root, spec, result, store)
    else:
        result["diagnostics"] = []
    if enable_feedback:
        progress.set_student_phase(student.student_id, "feedback")
        _attach_feedback(repo_root, spec, result, store)
    else:
        result["feedback"] = []
    store.write_student_result(student.student_id, result)
    progress.set_student_phase(student.student_id, "done")
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


def _run_selected_tests(
    repo_root: Path,
    spec,
    student: SubmittedStudent,
    selection: TestSelection,
    result_dir: Path,
) -> tuple[DockerRunResult, dict[str, DockerRunResult]]:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    xml_parts: list[str] = []
    elapsed_sec = 0.0
    exit_code = 0
    per_test_results: dict[str, DockerRunResult] = {}

    for selected_test in selection.selected_tests:
        test_file = Path(selected_test).name
        test_result_dir = result_dir / "test_results" / Path(test_file).stem
        test_result_dir.mkdir(parents=True, exist_ok=True)
        docker_raw = run_student_tests(
            spec,
            student.student_id,
            student.files,
            [selected_test],
            test_result_dir,
            repo_root=repo_root,
            source_dir=student.source_dir,
        )
        docker_result = _coerce_docker_result(docker_raw, test_result_dir)
        per_test_results[test_file] = docker_result
        if docker_result.exit_code and exit_code == 0:
            exit_code = docker_result.exit_code
        elapsed_sec += docker_result.elapsed_sec
        if docker_result.stdout:
            stdout_parts.append(f"=== {test_file} stdout ===\n{docker_result.stdout.rstrip()}\n")
        if docker_result.stderr:
            stderr_parts.append(f"=== {test_file} stderr ===\n{docker_result.stderr.rstrip()}\n")
        if docker_result.pytest_xml.strip():
            xml_parts.append(docker_result.pytest_xml)

    return (
        DockerRunResult(
            exit_code=exit_code,
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            elapsed_sec=elapsed_sec,
            pytest_xml=_combine_pytest_xml(xml_parts),
        ),
        per_test_results,
    )


def _combine_pytest_xml(xml_parts: list[str]) -> str:
    roots: list[ElementTree.Element] = []
    for xml in xml_parts:
        try:
            roots.append(ElementTree.fromstring(xml))
        except ElementTree.ParseError:
            continue
    if not roots:
        return ""
    if len(roots) == 1:
        return ElementTree.tostring(roots[0], encoding="unicode")
    combined = ElementTree.Element("testsuites")
    for root in roots:
        combined.append(root)
    return ElementTree.tostring(combined, encoding="unicode")


def _fill_unreported_selected_problems(
    spec,
    selection: TestSelection,
    problems: list[ProblemRunResult],
    docker_result: DockerRunResult,
    per_test_results: dict[str, DockerRunResult] | None = None,
) -> list[ProblemRunResult]:
    seen = {problem.test_file for problem in problems}
    selected_files = {Path(path).name for path in selection.selected_tests}
    per_test_results = per_test_results or {}
    for test_file, problem_id in spec.problem_ids.items():
        if test_file not in selected_files or test_file in seen:
            continue
        test_result = per_test_results.get(test_file, docker_result)
        status = _unreported_status(spec, test_result)
        max_points = float(spec.points.get(problem_id, 0))
        problems.append(
            ProblemRunResult(
                problem_id=problem_id,
                test_file=test_file,
                status=status,
                points_possible=max_points,
                points_earned=max_points if status == "passed" else 0,
                message=_unreported_message(status),
            )
        )
    return problems


def _unreported_status(spec, docker_result: DockerRunResult) -> str:
    if docker_result.exit_code == 0:
        return "passed"
    if _looks_like_timeout(spec, docker_result):
        return "timeout"
    return "error"


def _looks_like_timeout(spec, docker_result: DockerRunResult) -> bool:
    if docker_result.exit_code in {124, 137, 143, -9, -15}:
        return True
    try:
        timeout_sec = float(spec.limits.get("timeout_sec", 0) or 0)
    except (TypeError, ValueError):
        timeout_sec = 0.0
    return bool(timeout_sec and docker_result.elapsed_sec >= max(1.0, timeout_sec - 1.0))


def _unreported_message(status: str) -> str:
    if status == "timeout":
        return "No JUnit testcase was reported before this selected test timed out."
    return "No JUnit testcase was reported for this selected test."


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


def _attach_feedback(
    repo_root: Path,
    spec,
    result: dict[str, Any],
    store: BatchArtifactStore,
) -> None:
    if result.get("status") == "passed":
        result.setdefault("feedback", [])
        return
    try:
        generate_feedback_for_student(
            repo_root=repo_root,
            spec=spec,
            student_result=result,
            artifact_store=store,
        )
    except Exception as exc:
        result.setdefault("feedback", [])
        result["feedback_error"] = str(exc)


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


class _ProgressWriter:
    """Thread-safe state writer for dashboard polling."""

    def __init__(
        self,
        store: BatchArtifactStore,
        *,
        started_at: str,
        counts: dict[str, int],
    ) -> None:
        self.store = store
        self.started_at = started_at
        self.counts = counts
        self.student_phases: dict[str, str] = {}
        self.active_jobs: list[str] = []
        self._lock = threading.Lock()

    def set_student_phase(self, student_id: str, phase: str) -> None:
        with self._lock:
            self.student_phases[student_id] = phase
            self._write_locked("running")

    def write(
        self,
        status: str,
        *,
        active_jobs: list[str] | None = None,
        finished_at: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._write_locked(status, active_jobs=active_jobs, finished_at=finished_at, error=error)

    def _write_locked(
        self,
        status: str,
        *,
        active_jobs: list[str] | None = None,
        finished_at: str | None = None,
        error: str | None = None,
    ) -> None:
        if active_jobs is not None:
            self.active_jobs = list(active_jobs)
        self.store.update_state(
            {
                "status": status,
                "counts": dict(self.counts),
                "started_at": self.started_at,
                "finished_at": finished_at,
                "active_jobs": list(self.active_jobs),
                "students": dict(sorted(self.student_phases.items())),
                "pid": os_getpid(),
                "updated_at": _now(),
                "error": error,
            }
        )


def os_getpid() -> int:
    # Kept as a tiny wrapper so tests can monkeypatch it without touching os.
    import os

    return os.getpid()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reports-only local batch grading.")
    parser.add_argument("--homework", required=True)
    parser.add_argument("--submissions-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args(argv)

    try:
        run_batch(
            args.homework,
            args.submissions_root,
            args.output_root,
            args.run_id,
            max_workers=args.max_workers,
            enable_diagnostics=not args.no_diagnostics,
            enable_feedback=not args.no_feedback,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Batch setup failed: {exc}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
