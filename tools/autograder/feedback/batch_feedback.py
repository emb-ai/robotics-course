"""Batch orchestration for TA-review-only feedback draft artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import FeedbackClientConfig, FeedbackResult, generate_feedback
from .prompt_builder import FeedbackPromptContext, build_feedback_prompt


FEEDBACK_PROBLEM_STATUSES = {"failed", "error", "timeout", "missing", "skipped"}


def generate_feedback_for_student(
    repo_root: str | Path,
    spec: Any,
    student_result: dict[str, Any],
    artifact_store: Any,
    config: FeedbackClientConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate feedback draft artifacts for every non-passed problem."""

    refs: list[dict[str, Any]] = []
    feedback_entries = student_result.setdefault("feedback", [])
    artifacts = student_result.setdefault("artifacts", [])
    for problem_id, problem_result in (student_result.get("problems") or {}).items():
        if str(problem_result.get("status", "")).lower() not in FEEDBACK_PROBLEM_STATUSES:
            continue
        try:
            prompt = build_feedback_prompt(
                FeedbackPromptContext(
                    repo_root=repo_root,
                    spec=spec,
                    student_result=student_result,
                    problem_id=str(problem_id),
                    problem_result=problem_result,
                    artifact_store=artifact_store,
                )
            )
            result = generate_feedback(prompt, config=config)
            metadata = _metadata(student_result, str(problem_id), prompt.metadata, result)
            markdown = _markdown(result, metadata)
        except Exception as exc:
            result = FeedbackResult(status="error", error=str(exc))
            metadata = _metadata(student_result, str(problem_id), {}, result)
            markdown = _markdown(result, metadata)

        problem_refs = artifact_store.write_feedback(
            str(student_result.get("student_id", "")),
            str(problem_id),
            markdown,
            metadata,
        )
        refs.extend(problem_refs)
        artifacts.extend(problem_refs)
        feedback_entries.append(
            {
                "problem_id": str(problem_id),
                "status": result.status,
                "error": result.error,
                "artifacts": problem_refs,
            }
        )
    return refs


def _metadata(
    student_result: dict[str, Any],
    problem_id: str,
    prompt_metadata: dict[str, Any],
    result: FeedbackResult,
) -> dict[str, Any]:
    diagnostics = _diagnostic_summaries(student_result, problem_id)
    model = result.model or ""
    manifest = {
        "student_id": str(student_result.get("student_id", "")),
        "problem_id": problem_id,
        "student_files": list(prompt_metadata.get("student_files", [])),
        "reference_files": list(prompt_metadata.get("reference_files", [])),
        "diagnostics": diagnostics,
        "diagnostic_artifacts": list(prompt_metadata.get("diagnostic_artifacts", [])),
        "model": model,
        "prompt_token_budget_approx": int(prompt_metadata.get("prompt_token_budget_approx", 0) or 0),
        "timestamp": _now(),
    }
    return {
        "status": result.status,
        "error": result.error,
        "model": model,
        "prompt_token_budget_approx": manifest["prompt_token_budget_approx"],
        "manifest": manifest,
    }


def _markdown(result: FeedbackResult, metadata: dict[str, Any]) -> str:
    lines = [
        "# TA REVIEW ONLY - LLM Feedback Draft",
        "",
        "This draft is for TA review only. Do not send it directly to students without review.",
        "",
        f"Status: `{result.status}`",
    ]
    if metadata.get("model"):
        lines.append(f"Model: `{metadata['model']}`")
    lines.append("")
    if result.status == "ok":
        lines.extend(["## Draft", "", result.content.strip() or "(empty draft)", ""])
    else:
        lines.extend(["## Draft", "", "No draft content was generated.", ""])
        if result.error:
            lines.extend(["## Error", "", result.error, ""])
    lines.extend(["## Source Manifest", ""])
    manifest = metadata.get("manifest", {})
    lines.extend(
        [
            f"- Student: `{manifest.get('student_id', '')}`",
            f"- Problem: `{manifest.get('problem_id', '')}`",
            f"- Student files: {_format_list(manifest.get('student_files', []))}",
            f"- Reference files: {_format_list(manifest.get('reference_files', []))}",
            f"- Diagnostics: {_format_list(manifest.get('diagnostics', []))}",
            f"- Diagnostic artifacts: {_format_list(manifest.get('diagnostic_artifacts', []))}",
            f"- Prompt token budget approximation: `{manifest.get('prompt_token_budget_approx', 0)}`",
            f"- Timestamp: `{manifest.get('timestamp', '')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostic_summaries(student_result: dict[str, Any], problem_id: str) -> list[str]:
    summaries = []
    for diagnostic in student_result.get("diagnostics") or []:
        if not isinstance(diagnostic, dict):
            continue
        diagnostic_problem = str(diagnostic.get("problem_id", ""))
        if diagnostic_problem and diagnostic_problem != problem_id:
            continue
        plugin = str(diagnostic.get("plugin_id", "diagnostic"))
        summary = str(diagnostic.get("summary", "")).strip()
        summaries.append(f"{plugin}: {summary}".rstrip())
    return summaries


def _format_list(values: Any) -> str:
    items = [str(value) for value in values or [] if str(value)]
    if not items:
        return "`(none)`"
    return ", ".join(f"`{item}`" for item in items)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
