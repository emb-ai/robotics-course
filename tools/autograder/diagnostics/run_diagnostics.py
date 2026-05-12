"""Orchestration helper for running diagnostics on one student result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DiagnosticContext
from .generic import default_plugins
from .registry import DiagnosticRegistry, should_run_diagnostics


def run_diagnostics_for_student(
    repo_root: str | Path,
    spec: Any,
    student_result: dict[str, Any],
    artifact_store: Any,
    registry: DiagnosticRegistry | None = None,
) -> list[dict[str, Any]]:
    """Run fail-soft diagnostics for every non-passed problem in a student result."""

    active_registry = registry or DiagnosticRegistry(default_plugins())
    student_id = str(student_result["student_id"])
    submitted_files = list(student_result.get("submitted_files") or [])
    submitted_dir = artifact_store.student_dir(student_id) / "submitted"
    results: list[dict[str, Any]] = []
    for problem_id, problem_result in (student_result.get("problems") or {}).items():
        if not should_run_diagnostics(problem_result):
            continue
        context = DiagnosticContext(
            repo_root=Path(repo_root),
            run_dir=artifact_store.run_dir,
            topic_slug=str(_get(spec, "topic_slug", "")),
            homework_id=str(_get(spec, "id", "")),
            student_id=student_id,
            problem_id=str(problem_id),
            test_file=str(problem_result.get("test_file", "")),
            submitted_files=submitted_files,
            normalized_submission_dir=submitted_dir,
            student_result=student_result,
            problem_result=problem_result,
            artifact_store=artifact_store,
        )
        results.extend(result.to_dict() for result in active_registry.run(context))
    return results


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
