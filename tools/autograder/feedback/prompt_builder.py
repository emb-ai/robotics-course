"""Prompt assembly for TA-review-only feedback drafts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are drafting feedback for a TA in an AI in Robotics course. Use the problem "
    "statement, tests, reference snippets, student code, and diagnostics to produce a "
    "concise technical draft. Do not invent results. Do not write a full solution. "
    "Flag uncertainty. Mention which diagnostic artifacts are useful for review."
)

_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*['\"]?[^'\"\s]+"
)


@dataclass(frozen=True)
class FeedbackPrompt:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass(frozen=True)
class FeedbackPromptContext:
    repo_root: str | Path
    spec: Any
    student_result: dict[str, Any]
    problem_id: str
    problem_result: dict[str, Any]
    artifact_store: Any
    section_char_budget: int = 4000


def build_feedback_prompt(context: FeedbackPromptContext) -> FeedbackPrompt:
    """Build a deterministic LLM prompt for one failed/incomplete problem."""

    repo_root = Path(context.repo_root)
    spec = context.spec
    student_id = str(context.student_result.get("student_id", ""))
    problem_id = str(context.problem_id)
    problem_result = context.problem_result or {}
    test_file = str(problem_result.get("test_file") or _test_file_for_problem(spec, problem_id))
    solution_files = _solution_files_for_problem(spec, test_file, context.student_result)
    diagnostics = _diagnostics_for_problem(context.student_result, problem_id)
    artifacts = _diagnostic_artifact_labels(diagnostics)

    sections = [
        _section("Problem Metadata", _problem_metadata(spec, student_id, problem_id, test_file, problem_result), context),
        _section("Problem Text", _load_homework_text(repo_root, spec), context),
        _section("Optional Hints", _load_hints(repo_root, spec, problem_id), context),
        _section("Public Test Output", _public_test_output(problem_result), context),
        _section("Diagnostics", _format_diagnostics(diagnostics), context),
        _section("Diagnostic Artifacts", _format_artifacts(diagnostics), context),
        _section("Student Code", _load_student_files(context, student_id, solution_files), context),
        _section("Reference Snippets", _load_reference_files(repo_root, spec, solution_files), context),
        _section(
            "Drafting Policy",
            (
                "Write for the TA, not directly for the student. Identify likely weak points, "
                "explain relevant concepts, cite diagnostic artifacts by label, and suggest "
                "concise improvement directions. Do not claim hidden-test certainty beyond "
                "the diagnostics shown here."
            ),
            context,
        ),
    ]
    user_prompt = _redact("\n\n".join(sections).strip() + "\n")
    metadata = {
        "student_id": student_id,
        "problem_id": problem_id,
        "test_file": test_file,
        "student_files": _existing_student_files(context, student_id, solution_files),
        "reference_files": _existing_reference_files(repo_root, spec, solution_files),
        "diagnostic_artifacts": artifacts,
        "prompt_token_budget_approx": _approx_tokens(SYSTEM_PROMPT + "\n" + user_prompt),
    }
    return FeedbackPrompt(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, metadata=metadata)


def _section(title: str, content: str, context: FeedbackPromptContext) -> str:
    body = content.strip() if content and content.strip() else "(none)"
    return f"## {title}\n\n{_truncate(_redact(body), context.section_char_budget)}"


def _truncate(text: str, budget: int) -> str:
    if budget <= 0 or len(text) <= budget:
        return text
    marker = "\n...[truncated]"
    keep = max(0, budget - len(marker))
    return text[:keep].rstrip() + marker


def _redact(text: str) -> str:
    return _SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _topic_slug(spec: Any) -> str:
    return str(_get(spec, "topic_slug", ""))


def _problem_metadata(
    spec: Any,
    student_id: str,
    problem_id: str,
    test_file: str,
    problem_result: dict[str, Any],
) -> str:
    points = problem_result.get("max_points")
    if points is None:
        points = (_get(spec, "points", {}) or {}).get(problem_id, "")
    lines = [
        f"Homework: {_get(spec, 'id', '')}",
        f"Topic: {_topic_slug(spec)}",
        f"Student: {student_id}",
        f"Problem: {problem_id}",
        f"Test file: {test_file}",
        f"Status: {problem_result.get('status', '')}",
        f"Points: {problem_result.get('points', '')}/{points}",
    ]
    return "\n".join(lines)


def _public_test_output(problem_result: dict[str, Any]) -> str:
    fields = {
        "status": problem_result.get("status", ""),
        "message": problem_result.get("message", ""),
        "points": problem_result.get("points", ""),
        "max_points": problem_result.get("max_points", ""),
    }
    return json.dumps(fields, ensure_ascii=False, indent=2, sort_keys=True)


def _load_homework_text(repo_root: Path, spec: Any) -> str:
    nb_path = Path(_get(spec, "homework_dir", "")) / "homework.ipynb"
    if not nb_path.is_absolute():
        nb_path = repo_root / _topic_slug(spec) / "homework" / "homework.ipynb"
    if not nb_path.is_file():
        return ""
    try:
        import nbformat

        nb = nbformat.read(str(nb_path), as_version=4)
        cells = nb.cells
    except Exception:
        try:
            raw = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        cells = raw.get("cells", [])

    parts: list[str] = []
    for cell in cells:
        cell_type = _get(cell, "cell_type", "")
        if cell_type not in {"markdown", "code"}:
            continue
        source = _get(cell, "source", "")
        if isinstance(source, list):
            source = "".join(source)
        if str(source).strip():
            parts.append(str(source).strip())
    return "\n\n".join(parts)


def _load_hints(repo_root: Path, spec: Any, problem_id: str) -> str:
    path = repo_root / "dev" / _topic_slug(spec) / "homework" / "feedback_hints" / f"{problem_id}.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _test_file_for_problem(spec: Any, problem_id: str) -> str:
    for test_file, mapped_problem in (_get(spec, "problem_ids", {}) or {}).items():
        if str(mapped_problem) == problem_id:
            return str(test_file)
    return ""


def _solution_files_for_problem(spec: Any, test_file: str, student_result: dict[str, Any]) -> list[str]:
    dependencies = _get(spec, "test_dependencies", {}) or {}
    files = dependencies.get(test_file) if isinstance(dependencies, dict) else None
    if not files:
        files = student_result.get("submitted_files") or _get(spec, "solution_files", []) or []
    return sorted({str(name) for name in files if str(name)})


def _diagnostics_for_problem(student_result: dict[str, Any], problem_id: str) -> list[dict[str, Any]]:
    diagnostics = []
    for item in student_result.get("diagnostics") or []:
        if not isinstance(item, dict):
            continue
        item_problem = str(item.get("problem_id", ""))
        if item_problem == problem_id or not item_problem:
            diagnostics.append(item)
    return diagnostics


def _format_diagnostics(diagnostics: list[dict[str, Any]]) -> str:
    if not diagnostics:
        return ""
    lines = []
    for diagnostic in diagnostics:
        plugin = str(diagnostic.get("plugin_id", "diagnostic"))
        status = str(diagnostic.get("status", ""))
        summary = str(diagnostic.get("summary", ""))
        error = str(diagnostic.get("error") or "")
        line = f"- {plugin} [{status}]: {summary}".rstrip()
        if error:
            line += f" Error: {error}"
        lines.append(line)
    return "\n".join(lines)


def _format_artifacts(diagnostics: list[dict[str, Any]]) -> str:
    labels = []
    for diagnostic in diagnostics:
        for artifact in diagnostic.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            label = str(artifact.get("label") or artifact.get("path") or "")
            path = str(artifact.get("path") or "")
            description = str(artifact.get("description") or "")
            labels.append(f"- {label}: {path} {description}".strip())
    return "\n".join(labels)


def _diagnostic_artifact_labels(diagnostics: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for diagnostic in diagnostics:
        for artifact in diagnostic.get("artifacts") or []:
            if isinstance(artifact, dict):
                label = str(artifact.get("label") or artifact.get("path") or "")
                if label:
                    labels.append(label)
    return labels


def _load_student_files(context: FeedbackPromptContext, student_id: str, solution_files: list[str]) -> str:
    root = context.artifact_store.student_dir(student_id) / "submitted"
    chunks = []
    for name in solution_files:
        path = root / name
        if path.is_file():
            chunks.append(_code_block(name, path.read_text(encoding="utf-8", errors="replace")))
    return "\n\n".join(chunks)


def _load_reference_files(repo_root: Path, spec: Any, solution_files: list[str]) -> str:
    root = repo_root / "dev" / _topic_slug(spec) / "homework" / "reference_solution"
    chunks = []
    for name in solution_files:
        path = root / name
        if path.is_file():
            chunks.append(_code_block(name, path.read_text(encoding="utf-8", errors="replace")))
    return "\n\n".join(chunks)


def _code_block(filename: str, source: str) -> str:
    return f"### {filename}\n\n```python\n{source.rstrip()}\n```"


def _existing_student_files(
    context: FeedbackPromptContext,
    student_id: str,
    solution_files: list[str],
) -> list[str]:
    root = context.artifact_store.student_dir(student_id) / "submitted"
    return [name for name in solution_files if (root / name).is_file()]


def _existing_reference_files(repo_root: Path, spec: Any, solution_files: list[str]) -> list[str]:
    root = repo_root / "dev" / _topic_slug(spec) / "homework" / "reference_solution"
    return [name for name in solution_files if (root / name).is_file()]
