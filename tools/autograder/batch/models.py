"""JSON-serializable data models for reports-only batch grading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ProblemStatus = Literal["passed", "failed", "error", "skipped", "missing", "timeout"]


@dataclass
class HomeworkSpec:
    id: str
    topic_slug: str
    homework_dir: str
    compose_file: str
    solution_files: list[str]
    problem_ids: dict[str, str]
    points: dict[str, float]
    metrics: dict[str, Any]
    limits: dict[str, Any]
    test_dependencies: dict[str, list[str]]


@dataclass
class SubmittedStudent:
    student_id: str
    source_dir: str
    files: dict[str, str]
    ignored_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ProblemRunResult:
    problem_id: str
    test_file: str
    status: ProblemStatus
    points_possible: float
    points_earned: float
    message: str = ""


@dataclass
class StudentRunResult:
    student_id: str
    status: str
    submitted_files: list[str]
    ignored_files: list[str]
    missing_files: list[str]
    selected_tests: list[str]
    problem_results: list[ProblemRunResult]
    metrics: dict[str, float] = field(default_factory=dict)
    exit_code: int | None = None
    elapsed_sec: float = 0.0
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BatchConfig:
    run_id: str
    homework_id: str
    submissions_root: str
    output_root: str
    max_workers: int
    created_at: str


@dataclass
class BatchState:
    status: str
    counts: dict[str, int]
    started_at: str | None = None
    finished_at: str | None = None
    active_jobs: list[str] = field(default_factory=list)
    error: str | None = None


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths to JSON-compatible primitives."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
