"""Small generic diagnostics that work across homeworks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .base import DiagnosticContext, DiagnosticPlugin, DiagnosticResult


class PytestFailureExcerptPlugin:
    id = "pytest_failure_excerpt"
    label = "Pytest failure excerpt"
    timeout_sec = 2.0

    def supports(self, context: DiagnosticContext) -> bool:
        return _problem_status(context) in {"failed", "error", "skipped"}

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        text = _student_log_text(context)
        excerpt = _failure_excerpt(text)
        if not excerpt:
            excerpt = _problem_message(context) or "No pytest failure excerpt was found."
        artifact = context.artifact_store.write_text_artifact(
            context.student_id,
            context.problem_id,
            "diagnostic",
            "failure_excerpt.md",
            f"# Failure excerpt\n\n```text\n{excerpt.strip()}\n```\n",
            label="failure_excerpt.md",
            description="First useful pytest failure or error excerpt.",
        )
        first_line = _first_failure_line(excerpt) or (
            excerpt.strip().splitlines()[0] if excerpt.strip() else "No failure excerpt found."
        )
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=first_line[:240],
            artifacts=[artifact],
        )


class StaticScanPlugin:
    id = "submission_static_scan"
    label = "Submission static scan"
    timeout_sec = 3.0

    def supports(self, context: DiagnosticContext) -> bool:
        return bool(context.submitted_files)

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        report = {
            "syntax_errors": [],
            "forbidden_imports": [],
            "files": sorted(context.submitted_files),
        }
        for filename in sorted(context.submitted_files):
            if not filename.endswith(".py"):
                continue
            path = context.normalized_submission_dir / filename
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source, filename=filename)
            except SyntaxError as exc:
                report["syntax_errors"].append(
                    {
                        "file": filename,
                        "line": exc.lineno,
                        "message": exc.msg,
                    }
                )
                continue
            report["forbidden_imports"].extend(_forbidden_imports(filename, tree))

        json_artifact = context.artifact_store.write_text_artifact(
            context.student_id,
            context.problem_id,
            "diagnostic",
            "static_scan.json",
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            label="static_scan.json",
            description="Machine-readable static scan report.",
        )
        markdown = _static_scan_markdown(report)
        md_artifact = context.artifact_store.write_text_artifact(
            context.student_id,
            context.problem_id,
            "diagnostic",
            "static_scan.md",
            markdown,
            label="static_scan.md",
            description="Human-readable static scan report.",
        )
        syntax_count = len(report["syntax_errors"])
        import_count = len(report["forbidden_imports"])
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=f"Static scan: {syntax_count} syntax error(s), {import_count} forbidden import(s).",
            metrics={"syntax_errors": syntax_count, "forbidden_imports": import_count},
            artifacts=[json_artifact, md_artifact],
        )


class TimeoutClassifierPlugin:
    id = "timeout_classifier"
    label = "Timeout classifier"
    timeout_sec = 2.0

    def supports(self, context: DiagnosticContext) -> bool:
        text = f"{_problem_status(context)} {_problem_message(context)} {_student_log_text(context)}"
        return "timeout" in text.lower() or "timed out" in text.lower()

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        text = _student_log_text(context)
        hints = []
        lowered = text.lower()
        if "timeout" in lowered or "timed out" in lowered:
            hints.append("The grader reported a timeout.")
        if "while" in lowered or "for " in lowered:
            hints.append("Look for loops or simulations that may not terminate quickly.")
        if not hints:
            hints.append("The problem result is timeout-like, but logs did not include a detailed cause.")
        body = "# Timeout diagnostic\n\n" + "\n".join(f"- {hint}" for hint in hints) + "\n"
        artifact = context.artifact_store.write_text_artifact(
            context.student_id,
            context.problem_id,
            "diagnostic",
            "timeout.md",
            body,
            label="timeout.md",
            description="Generic timeout classification.",
        )
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=hints[0],
            artifacts=[artifact],
        )


class MissingDependenciesPlugin:
    id = "missing_dependencies"
    label = "Missing dependencies"
    timeout_sec = 2.0

    def supports(self, context: DiagnosticContext) -> bool:
        return _problem_status(context) == "missing"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        expected = _missing_files(context)
        submitted = sorted(context.student_result.get("submitted_files") or context.submitted_files)
        lines = [
            "# Missing dependencies",
            "",
            "## Expected",
            *[f"- `{name}`" for name in expected],
            "",
            "## Submitted",
            *[f"- `{name}`" for name in submitted],
            "",
        ]
        artifact = context.artifact_store.write_text_artifact(
            context.student_id,
            context.problem_id,
            "diagnostic",
            "missing_dependencies.md",
            "\n".join(lines),
            label="missing_dependencies.md",
            description="Missing dependency report.",
        )
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=f"Missing dependencies: {', '.join(expected) if expected else 'unknown'}.",
            metrics={"missing_files": len(expected)},
            artifacts=[artifact],
        )


def default_plugins() -> list[DiagnosticPlugin]:
    from .hw01 import default_plugins as hw01_plugins
    from .hw02 import default_plugins as hw02_plugins
    from .hw03 import default_plugins as hw03_plugins

    return [
        PytestFailureExcerptPlugin(),
        StaticScanPlugin(),
        TimeoutClassifierPlugin(),
        MissingDependenciesPlugin(),
        *hw01_plugins(),
        *hw02_plugins(),
        *hw03_plugins(),
    ]


def _student_log_text(context: DiagnosticContext) -> str:
    student_dir = context.artifact_store.student_dir(context.student_id)
    chunks = []
    for name in ("stdout.log", "stderr.log"):
        path = student_dir / name
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _failure_excerpt(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _is_failure_line(line):
            start = max(0, index - 2)
            end = min(len(lines), index + max_lines)
            return "\n".join(lines[start:end])
    return "\n".join(lines[:max_lines])


def _first_failure_line(text: str) -> str | None:
    for line in text.splitlines():
        if _is_failure_line(line):
            return line
    return None


def _is_failure_line(line: str) -> bool:
    lowered = line.lower()
    return (
        line.startswith("FAILED ")
        or " failed" in lowered
        or "error " in lowered
        or "traceback" in lowered
        or "assertionerror" in lowered
    )


def _forbidden_imports(filename: str, tree: ast.AST) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        for module in modules:
            if module == "reference_solution" or module.startswith("reference_solution."):
                findings.append({"file": filename, "line": getattr(node, "lineno", None), "module": module})
            if module == "hidden_tests" or module.startswith("hidden_tests."):
                findings.append({"file": filename, "line": getattr(node, "lineno", None), "module": module})
    return findings


def _static_scan_markdown(report: dict[str, Any]) -> str:
    lines = ["# Static scan", ""]
    lines.append(f"Files scanned: {len(report['files'])}")
    lines.append(f"Syntax errors: {len(report['syntax_errors'])}")
    for item in report["syntax_errors"]:
        lines.append(f"- `{item['file']}` line {item['line']}: {item['message']}")
    lines.append(f"Forbidden imports: {len(report['forbidden_imports'])}")
    for item in report["forbidden_imports"]:
        lines.append(f"- `{item['file']}` imports `{item['module']}`")
    lines.append("")
    return "\n".join(lines)


def _problem_status(context: DiagnosticContext) -> str:
    return str(context.problem_result.get("status", "")).lower()


def _problem_message(context: DiagnosticContext) -> str:
    return str(context.problem_result.get("message", ""))


def _missing_files(context: DiagnosticContext) -> list[str]:
    message = _problem_message(context)
    if ":" in message:
        tail = message.split(":", 1)[1]
        return [item.strip() for item in re.split(r",|\s+", tail) if item.strip().endswith(".py")]
    return sorted(context.student_result.get("missing_files") or [])
