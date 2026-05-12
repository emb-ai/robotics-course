"""Shared diagnostics data models."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


DiagnosticStatus = Literal["ok", "skipped", "error", "timeout"]


@dataclass
class DiagnosticContext:
    repo_root: Path
    run_dir: Path
    topic_slug: str
    homework_id: str
    student_id: str
    problem_id: str
    test_file: str
    submitted_files: list[str]
    normalized_submission_dir: Path
    student_result: dict[str, Any]
    problem_result: dict[str, Any]
    artifact_store: Any


@dataclass
class DiagnosticResult:
    plugin_id: str
    problem_id: str
    status: DiagnosticStatus
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


class DiagnosticPlugin(Protocol):
    id: str
    label: str
    timeout_sec: float

    def supports(self, context: DiagnosticContext) -> bool:
        ...

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
