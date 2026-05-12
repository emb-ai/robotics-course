"""Artifact storage for local batch grading runs."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import html
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_STUDENT_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


class BatchArtifactStore:
    """Safe writer for one local batch grading run."""

    def __init__(self, output_root: str | Path, run_id: str):
        self.output_root = Path(output_root)
        self.run_id = _validate_safe_segment(run_id, "run_id")
        self.run_dir = self.output_root / self.run_id
        self.students_root = self.run_dir / "students"
        self.students_root.mkdir(parents=True, exist_ok=True)
        self.update_state({"status": "created"})

    def write_config(self, config: Any) -> Path:
        """Write the immutable run config."""

        return self._atomic_write_json(self.run_dir / "config.json", config)

    def update_state(self, state: Any) -> Path:
        """Atomically update dashboard polling state."""

        return self._atomic_write_json(self.run_dir / "state.json", state)

    def student_dir(self, student_id: str) -> Path:
        """Return the safe per-student directory, creating metadata lazily."""

        display_id = _validate_student_id(student_id)
        path_id = self.student_path_id(display_id)
        directory = self.students_root / path_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "submitted").mkdir(exist_ok=True)
        (directory / "diagnostics").mkdir(exist_ok=True)
        (directory / "feedback").mkdir(exist_ok=True)
        self._atomic_write_json(
            directory / "student.json",
            {"student_id": display_id, "student_path_id": path_id},
        )
        return directory

    def student_path_id(self, student_id: str) -> str:
        """Return a filesystem-safe path id for a display student id."""

        display_id = _validate_student_id(student_id)
        base = _student_slug(display_id)
        candidate_dir = self.students_root / base
        metadata_path = candidate_dir / "student.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
            if metadata.get("student_id") == display_id:
                return base
            return f"{base}-{_short_hash(display_id)}"
        return base

    def write_student_result(self, student_id: str, result: Any) -> Path:
        """Write `students/<student>/result.json` with display metadata."""

        display_id = _validate_student_id(student_id)
        directory = self.student_dir(display_id)
        payload = _to_jsonable(result)
        if not isinstance(payload, dict):
            payload = {"result": payload}
        payload.setdefault("student_id", display_id)
        payload.setdefault("student_path_id", directory.name)
        return self._atomic_write_json(directory / "result.json", payload)

    def write_student_output(
        self,
        student_id: str,
        stdout: str = "",
        stderr: str = "",
        pytest_xml: str | bytes | None = None,
    ) -> dict[str, Path]:
        """Write root-level stdout, stderr, and optional JUnit XML artifacts."""

        directory = self.student_dir(student_id)
        written = {
            "stdout": self._atomic_write_text(directory / "stdout.log", stdout),
            "stderr": self._atomic_write_text(directory / "stderr.log", stderr),
        }
        if pytest_xml is not None:
            path = directory / "pytest.xml"
            if isinstance(pytest_xml, bytes):
                written["pytest_xml"] = self._atomic_write_bytes(path, pytest_xml)
            else:
                written["pytest_xml"] = self._atomic_write_text(path, pytest_xml)
        return written

    def write_submitted_file(
        self,
        student_id: str,
        relative_path: str | Path,
        data: str | bytes,
    ) -> Path:
        """Write one normalized submitted file under `submitted/`."""

        submitted_root = self.student_dir(student_id) / "submitted"
        dest = submitted_root / _validate_relative_path(relative_path)
        self._ensure_inside(submitted_root, dest)
        if isinstance(data, bytes):
            return self._atomic_write_bytes(dest, data)
        return self._atomic_write_text(dest, data)

    def copy_submitted_file(
        self,
        student_id: str,
        relative_path: str | Path,
        source_path: str | Path,
    ) -> Path:
        """Copy one submitted file while preventing destination traversal."""

        data = Path(source_path).read_bytes()
        return self.write_submitted_file(student_id, relative_path, data)

    def write_text_artifact(
        self,
        student_id: str,
        problem_id: str,
        kind: str,
        filename: str | Path,
        text: str,
        *,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Write a UTF-8 artifact and register its portable reference."""

        path = self._problem_artifact_path(student_id, problem_id, kind, filename)
        self._atomic_write_text(path, text)
        return self.register_artifact_ref(
            student_id,
            problem_id,
            kind,
            label or Path(str(filename)).name,
            path,
            description,
        )

    def write_bytes_artifact(
        self,
        student_id: str,
        problem_id: str,
        kind: str,
        filename: str | Path,
        data: bytes,
        *,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Write a binary artifact and register its portable reference."""

        path = self._problem_artifact_path(student_id, problem_id, kind, filename)
        self._atomic_write_bytes(path, data)
        return self.register_artifact_ref(
            student_id,
            problem_id,
            kind,
            label or Path(str(filename)).name,
            path,
            description,
        )

    def write_feedback(
        self,
        student_id: str,
        problem_id: str,
        markdown: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Write feedback markdown and JSON draft files for one problem."""

        display_id = _validate_student_id(student_id)
        safe_problem_id = _validate_safe_segment(problem_id, "problem_id")
        feedback_dir = self.student_dir(display_id) / "feedback"
        md_path = self._atomic_write_text(
            feedback_dir / f"{safe_problem_id}.md",
            markdown,
        )
        json_path = self._atomic_write_json(
            feedback_dir / f"{safe_problem_id}.json",
            {
                "student_id": display_id,
                "student_path_id": self.student_path_id(display_id),
                "problem_id": problem_id,
                "feedback": markdown,
                "metadata": dict(metadata or {}),
            },
        )
        return [
            self.register_artifact_ref(
                display_id,
                problem_id,
                "feedback",
                f"{problem_id}.md",
                md_path,
                "LLM feedback draft markdown.",
            ),
            self.register_artifact_ref(
                display_id,
                problem_id,
                "feedback",
                f"{problem_id}.json",
                json_path,
                "LLM feedback draft metadata.",
            ),
        ]

    def register_artifact_ref(
        self,
        student_id: str,
        problem_id: str,
        kind: str,
        label: str,
        path: str | Path,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Register an artifact reference if the path is inside this run."""

        display_id = _validate_student_id(student_id)
        student_directory = self.student_dir(display_id)
        resolved = self._resolve_inside_run(path)
        rel = resolved.relative_to(self.run_dir.resolve()).as_posix()
        ref: dict[str, Any] = {
            "kind": str(kind),
            "label": str(label),
            "path": rel,
            "problem_id": problem_id,
        }
        if description:
            ref["description"] = description

        index_path = student_directory / "artifacts.json"
        existing = self._read_json(index_path, default={})
        artifacts = list(existing.get("artifacts", [])) if isinstance(existing, dict) else []
        artifacts.append(ref)
        self._atomic_write_json(
            index_path,
            {
                "student_id": display_id,
                "student_path_id": student_directory.name,
                "artifacts": artifacts,
            },
        )
        return ref

    def write_results(self, batch_result: Any) -> Path:
        """Write the full structured batch result."""

        return self._atomic_write_json(self.run_dir / "results.json", batch_result)

    def write_summary_csv(self, batch_result: Any) -> Path:
        """Write one CSV row per student/problem, including missing problems."""

        rows = self._summary_rows(batch_result)
        path = self.run_dir / "summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            writer = csv.DictWriter(
                tmp,
                fieldnames=[
                    "student_id",
                    "student_path_id",
                    "problem_id",
                    "status",
                    "points",
                    "max_points",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
        return path

    def write_index_html(self, batch_result: Any) -> Path:
        """Write a static report with links to available artifacts."""

        rows = self._summary_rows(batch_result)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["student_id"], []).append(row)

        body = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Batch grading report</title>",
            "<style>",
            "body{font-family:sans-serif;margin:24px;max-width:1200px}",
            "table{border-collapse:collapse;width:100%;margin-bottom:24px}",
            "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}",
            "th{background:#eee}",
            "code{background:#f5f5f5;padding:1px 3px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Batch grading report</h1>",
            '<p><a href="summary.csv">summary.csv</a> ',
            '<a href="results.json">results.json</a> ',
            '<a href="state.json">state.json</a></p>',
        ]
        for student_id, student_rows in grouped.items():
            path_id = student_rows[0]["student_path_id"]
            body.extend(
                [
                    f"<h2>{html.escape(student_id)}</h2>",
                    f"<p><code>students/{html.escape(path_id)}</code></p>",
                    "<table>",
                    "<thead><tr>",
                    "<th>Problem</th><th>Status</th><th>Points</th><th>Artifacts</th>",
                    "</tr></thead>",
                    "<tbody>",
                ]
            )
            refs_by_problem = self._artifact_refs_by_problem(student_id)
            for row in student_rows:
                problem_id = row["problem_id"]
                refs = refs_by_problem.get(problem_id, [])
                links = " ".join(_artifact_link(ref) for ref in refs) or ""
                body.append(
                    "<tr>"
                    f"<td>{html.escape(problem_id)}</td>"
                    f"<td>{html.escape(str(row['status']))}</td>"
                    f"<td>{html.escape(str(row['points']))}/"
                    f"{html.escape(str(row['max_points']))}</td>"
                    f"<td>{links}</td>"
                    "</tr>"
                )
            body.extend(["</tbody>", "</table>"])
        body.extend(["</body>", "</html>", ""])
        return self._atomic_write_text(self.run_dir / "index.html", "\n".join(body))

    def _problem_artifact_path(
        self,
        student_id: str,
        problem_id: str,
        kind: str,
        filename: str | Path,
    ) -> Path:
        safe_problem_id = _validate_safe_segment(problem_id, "problem_id")
        relative = _validate_relative_path(filename)
        if kind == "feedback":
            base = self.student_dir(student_id) / "feedback"
        else:
            base = self.student_dir(student_id) / "diagnostics" / safe_problem_id
        path = base / relative
        self._ensure_inside(base, path)
        return path

    def _summary_rows(self, batch_result: Any) -> list[dict[str, Any]]:
        students = _extract_students(batch_result)
        problem_ids = _extract_problem_ids(batch_result, students)
        point_map = _extract_points(batch_result)
        rows: list[dict[str, Any]] = []
        for student in students:
            student_id = _validate_student_id(_student_id(student))
            path_id = self.student_dir(student_id).name
            problems = _student_problems(student)
            for problem_id in problem_ids:
                problem = problems.get(problem_id)
                max_points = _problem_max_points(problem, point_map.get(problem_id, ""))
                rows.append(
                    {
                        "student_id": student_id,
                        "student_path_id": path_id,
                        "problem_id": problem_id,
                        "status": _problem_status(problem),
                        "points": _problem_points(problem),
                        "max_points": max_points,
                    }
                )
        return rows

    def _artifact_refs_by_problem(self, student_id: str) -> dict[str, list[dict[str, Any]]]:
        path_id = self.student_path_id(student_id)
        index = self._read_json(self.students_root / path_id / "artifacts.json", default={})
        refs = index.get("artifacts", []) if isinstance(index, dict) else []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            problem_id = str(ref.get("problem_id", ""))
            grouped.setdefault(problem_id, []).append(ref)
        return grouped

    def _atomic_write_json(self, path: Path, data: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(_to_jsonable(data), tmp, ensure_ascii=False, indent=2, sort_keys=True)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        return path

    def _atomic_write_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(text)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        return path

    def _atomic_write_bytes(self, path: Path, data: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        return path

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default

    def _resolve_inside_run(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.run_dir / candidate
        return self._ensure_inside(self.run_dir, candidate)

    @staticmethod
    def _ensure_inside(root: Path, path: Path) -> Path:
        root_resolved = root.resolve()
        path_resolved = path.resolve(strict=False)
        try:
            path_resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"path escapes artifact root: {path}") from exc
        return path_resolved


def _validate_safe_segment(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not _SAFE_SEGMENT_RE.fullmatch(value):
        raise ValueError(f"unsafe {field}: {value!r}")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def _validate_student_id(student_id: str) -> str:
    if not isinstance(student_id, str):
        raise TypeError("student_id must be a string")
    display_id = student_id.strip()
    if not display_id or display_id in {".", ".."}:
        raise ValueError(f"unsafe student_id: {student_id!r}")
    if "/" in display_id or "\\" in display_id or "\x00" in display_id:
        raise ValueError(f"unsafe student_id: {student_id!r}")
    return display_id


def _student_slug(student_id: str) -> str:
    slug = _SAFE_STUDENT_CHARS_RE.sub("_", student_id.strip()).strip("._-")
    if not slug:
        raise ValueError(f"unsafe student_id: {student_id!r}")
    return slug[:120]


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _validate_relative_path(path: str | Path) -> Path:
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"unsafe artifact path: {path!r}")
    raw = raw.replace("\\", "/")
    posix = PurePosixPath(raw)
    if posix.is_absolute():
        raise ValueError(f"unsafe artifact path: {path!r}")
    parts = posix.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe artifact path: {path!r}")
    if any(":" in part for part in parts):
        raise ValueError(f"unsafe artifact path: {path!r}")
    return Path(*parts)


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_students(batch_result: Any) -> list[Any]:
    for key in ("students", "student_results"):
        students = _get(batch_result, key)
        if students is not None:
            return list(students)
    results = _get(batch_result, "results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        return list(results)
    if isinstance(batch_result, Sequence) and not isinstance(batch_result, (str, bytes)):
        return list(batch_result)
    return []


def _extract_points(batch_result: Any) -> dict[str, Any]:
    points = _get(batch_result, "points", {})
    if isinstance(points, Mapping):
        return {str(k): v for k, v in points.items()}
    return {}


def _extract_problem_ids(batch_result: Any, students: list[Any]) -> list[str]:
    problem_ids: list[str] = []
    for key in ("problem_ids", "problem_order"):
        values = _get(batch_result, key)
        if values:
            problem_ids.extend(str(v) for v in values)
    problems = _get(batch_result, "problems")
    if isinstance(problems, Mapping):
        problem_ids.extend(str(k) for k in problems)
    elif isinstance(problems, Sequence) and not isinstance(problems, (str, bytes)):
        for problem in problems:
            if isinstance(problem, str):
                problem_ids.append(problem)
            else:
                problem_ids.append(str(_get(problem, "problem_id", _get(problem, "id"))))
    problem_ids.extend(_extract_points(batch_result).keys())
    for student in students:
        problem_ids.extend(_student_problems(student).keys())
    return _dedupe([pid for pid in problem_ids if pid and pid != "None"])


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _student_id(student: Any) -> str:
    for key in ("student_id", "display_id", "id", "name"):
        value = _get(student, key)
        if value is not None:
            return str(value)
    raise ValueError(f"student result is missing a student id: {student!r}")


def _student_problems(student: Any) -> dict[str, Any]:
    for key in ("problems", "problem_results", "results"):
        problems = _get(student, key)
        if isinstance(problems, Mapping):
            return {str(k): v for k, v in problems.items()}
        if isinstance(problems, Sequence) and not isinstance(problems, (str, bytes)):
            result: dict[str, Any] = {}
            for item in problems:
                problem_id = _get(item, "problem_id", _get(item, "id", _get(item, "name")))
                if problem_id is not None:
                    result[str(problem_id)] = item
            return result
    return {}


def _problem_status(problem: Any) -> str:
    if problem is None:
        return "missing"
    status = _get(problem, "status")
    if status is not None:
        return str(status)
    passed = _get(problem, "passed")
    if isinstance(passed, bool):
        return "passed" if passed else "failed"
    return "unknown"


def _problem_points(problem: Any) -> Any:
    if problem is None:
        return 0
    for key in ("points", "score"):
        value = _get(problem, key)
        if value is not None:
            return value
    return 0


def _problem_max_points(problem: Any, default: Any) -> Any:
    if problem is not None:
        for key in ("max_points", "total_points"):
            value = _get(problem, key)
            if value is not None:
                return value
    return default


def _artifact_link(ref: Mapping[str, Any]) -> str:
    path = str(ref.get("path", ""))
    if not path:
        return ""
    label = str(ref.get("label") or ref.get("kind") or path)
    description = str(ref.get("description") or "")
    return (
        f'<a href="{html.escape(path, quote=True)}"'
        f' title="{html.escape(description, quote=True)}">'
        f"{html.escape(label)}</a>"
    )
