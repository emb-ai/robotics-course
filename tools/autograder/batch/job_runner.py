"""Dashboard-launched orchestration for local batch grading jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .discovery import get_homework
from .runner import run_batch


_GRADEABLE_SUFFIXES = {".py", ".zip"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RANDOM_SUFFIX_RE = re.compile(r"^[A-Za-z0-9]{6,12}$")


@dataclass(frozen=True)
class _PreparedCandidate:
    filename: str
    order: tuple[int, str, int, str]
    content: bytes
    source: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one dashboard batch grading job.")
    parser.add_argument("--job-config", required=True)
    args = parser.parse_args(argv)

    job_path = Path(args.job_config)
    config = json.loads(job_path.read_text(encoding="utf-8"))
    run_dir = job_path.parent
    output_root = Path(config["output_root"])
    run_id = str(config["run_id"])

    try:
        submissions_root = _resolve_submissions_root(config, run_dir, job_path)
        run_batch(
            homework_id=str(config["homework_id"]),
            submissions_root=submissions_root,
            output_root=output_root,
            run_id=run_id,
            max_workers=int(config.get("max_workers", 2)),
            enable_diagnostics=bool(config.get("enable_diagnostics", True)),
            enable_feedback=bool(config.get("enable_feedback", True)),
        )
    except subprocess.CalledProcessError as exc:
        error = str(exc)
        if exc.cmd == "dataschool download":
            error = f"DataSchool download failed with exit code {exc.returncode}."
        _write_state(run_dir, {"status": "error", "error": error, "finished_at": _now()})
        return exc.returncode or 1
    except Exception as exc:
        _write_state(run_dir, {"status": "error", "error": str(exc), "finished_at": _now()})
        return 1
    return 0


def _resolve_submissions_root(config: dict[str, Any], run_dir: Path, job_path: Path) -> Path:
    mode = str(config.get("source_mode", "local"))
    if mode == "local":
        return Path(str(config["submissions_root"]))
    if mode == "dataschool":
        download_root = Path(
            str(config.get("download_root") or _repo_root() / "dev" / "downloads" / "dataschool_submissions")
        )
        _write_state(run_dir, {"status": "downloading", "started_at": _now()})
        code = _run_dataschool_downloader(config, download_root, run_dir, job_path)
        if code != 0:
            _write_state(
                run_dir,
                {
                    "status": "error",
                    "error": f"DataSchool download failed with exit code {code}.",
                    "finished_at": _now(),
                },
            )
            raise subprocess.CalledProcessError(code, "dataschool download")
        _write_state(run_dir, {"status": "preparing", "started_at": _now()})
        return prepare_dataschool_submissions(download_root, run_dir, homework_id=str(config.get("homework_id") or ""))
    raise ValueError(f"unknown source_mode: {mode}")


def _run_dataschool_downloader(
    config: dict[str, Any],
    download_root: Path,
    run_dir: Path,
    job_path: Path,
) -> int:
    repo_root = _repo_root()
    script = _downloader_script(repo_root)
    cmd = [sys.executable, str(script)]
    if config.get("queue_url"):
        cmd.extend(["--queue-url", str(config["queue_url"])])
    else:
        for key in ("course", "assignments", "statuses", "sort"):
            if config.get(key):
                cmd.extend([f"--{key.replace('_', '-')}", str(config[key])])
    cmd.extend(["--out", str(download_root)])
    cookie_file = str(config.get("cookie_file") or "").strip()
    if cookie_file:
        cmd.extend(["--cookie-file", cookie_file])
    if config.get("limit"):
        cmd.extend(["--limit", str(config["limit"])])

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=_subprocess_env(repo_root),
    )
    _atomic_write_text(run_dir / "dataschool_download_stdout.log", proc.stdout)
    _atomic_write_text(run_dir / "dataschool_download_stderr.log", proc.stderr)
    _write_state(
        run_dir,
        {
            "status": "downloading" if proc.returncode == 0 else "error",
            "download": {
                "returncode": proc.returncode,
                "stdout_log": "dataschool_download_stdout.log",
                "stderr_log": "dataschool_download_stderr.log",
            },
            "job_config": str(job_path),
        },
    )
    return int(proc.returncode)


def prepare_dataschool_submissions(download_root: Path, run_dir: Path, homework_id: str | None = None) -> Path:
    """Copy the latest gradeable student file per expected solution into batch layout."""

    prepared_root = run_dir / "prepared_submissions"
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = download_root / "submissions.jsonl"
    students: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not manifest_path.is_file():
        warnings.append(f"missing DataSchool interactions file: {manifest_path}")
        _write_prepared_manifest(run_dir, students, warnings)
        return prepared_root

    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            interaction = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        assignment = str(interaction.get("assignment") or "")
        if homework_id and not _assignment_matches_homework(str(homework_id), assignment):
            student = str(interaction.get("student") or "unknown")
            warnings.append(f"{student}: skipped assignment {assignment!r} for homework {homework_id}")
            continue
        expected_files = _solution_files_for_interaction(homework_id, assignment)
        selected = _latest_gradeable_files(interaction, expected_files, warnings)
        if not selected:
            warnings.append(f"{interaction.get('student', 'unknown')}: no gradeable attempt targets")
            continue
        student_id = _prepared_student_id(interaction)
        student_dir = prepared_root / student_id
        student_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        file_sources: dict[str, dict[str, Any]] = {}
        for filename, candidate in sorted(selected.items()):
            (student_dir / filename).write_bytes(candidate.content)
            copied.append(filename)
            file_sources[filename] = candidate.source
        latest_source = max(selected.values(), key=lambda candidate: candidate.order).source
        students.append(
            {
                "student_id": student_id,
                "student": interaction.get("student"),
                "submission_id": interaction.get("submission_id"),
                "assignment": interaction.get("assignment"),
                "attempt_index": latest_source.get("attempt_index"),
                "created_at": latest_source.get("created_at"),
                "files": sorted(copied),
                "file_sources": file_sources,
            }
        )

    _write_prepared_manifest(run_dir, students, warnings)
    return prepared_root


def _latest_gradeable_files(
    interaction: dict[str, Any],
    expected_files: list[str] | None,
    warnings: list[str],
) -> dict[str, _PreparedCandidate]:
    selected: dict[str, _PreparedCandidate] = {}
    expected = set(expected_files or [])
    for item in interaction.get("gradeable_attempts", []):
        if not isinstance(item, dict) or not _is_gradeable_target(item):
            continue
        for candidate in _attachment_candidates(item, expected or None, warnings):
            current = selected.get(candidate.filename)
            if current is None or candidate.order > current.order:
                selected[candidate.filename] = candidate
    return selected


def _attachment_candidates(
    item: dict[str, Any],
    expected_files: set[str] | None,
    warnings: list[str],
) -> list[_PreparedCandidate]:
    target = Path(str(item.get("target") or ""))
    suffix = Path(str(item.get("filename") or target.name)).suffix.casefold()
    if suffix == ".py":
        filename = _match_expected_filename(
            str(item.get("filename") or target.name),
            str(item.get("canonical_filename") or ""),
            expected_files,
        )
        if not filename:
            return []
        return [
            _PreparedCandidate(
                filename=filename,
                order=_candidate_order(item, ""),
                content=target.read_bytes(),
                source=_candidate_source(item),
            )
        ]
    if suffix != ".zip":
        return []

    candidates: list[_PreparedCandidate] = []
    try:
        with ZipFile(target) as archive:
            for info in sorted(archive.infolist(), key=lambda entry: entry.filename):
                if info.is_dir() or _unsafe_archive_name(info.filename):
                    continue
                entry_name = Path(info.filename.replace("\\", "/")).name
                if Path(entry_name).suffix.casefold() != ".py":
                    continue
                filename = _match_expected_filename(entry_name, "", expected_files)
                if not filename:
                    continue
                candidates.append(
                    _PreparedCandidate(
                        filename=filename,
                        order=_candidate_order(item, info.filename),
                        content=archive.read(info),
                        source=_candidate_source(item, archive_entry=info.filename),
                    )
                )
    except BadZipFile as exc:
        warnings.append(f"{item.get('student', 'unknown')}: invalid zip attachment {target}: {exc}")
    return candidates


def _solution_files_for_interaction(homework_id: str | None, assignment: str) -> list[str] | None:
    resolved_homework_id = str(homework_id or "").strip()
    if not resolved_homework_id:
        resolved_homework_id = _homework_id_from_assignment(assignment)
    if not resolved_homework_id:
        return None
    try:
        return list(get_homework(_repo_root(), resolved_homework_id).solution_files)
    except ValueError:
        return None


def _homework_id_from_assignment(assignment: str) -> str:
    match = re.search(r"(?:hw|homework|дз|домашн\w*)\D*(\d+)", assignment.casefold())
    if not match:
        return ""
    return f"{int(match.group(1)):02d}"


def _match_expected_filename(raw_name: str, canonical_name: str, expected_files: set[str] | None) -> str | None:
    raw = Path(raw_name).name
    canonical = Path(canonical_name).name if canonical_name else ""
    if Path(raw).suffix.casefold() != ".py":
        return None
    if expected_files is None:
        if canonical and Path(canonical).suffix.casefold() == ".py":
            return _safe_filename(canonical)
        return _safe_filename(raw)

    for name in (raw, canonical):
        if name in expected_files:
            return name

    raw_path = Path(raw)
    for expected in sorted(expected_files, key=len, reverse=True):
        expected_path = Path(expected)
        if raw_path.suffix.casefold() != expected_path.suffix.casefold():
            continue
        prefix = f"{expected_path.stem}_"
        if raw_path.stem.startswith(prefix) and _looks_like_random_suffix(raw_path.stem[len(prefix) :]):
            return expected
    return None


def _looks_like_random_suffix(value: str) -> bool:
    return bool(_RANDOM_SUFFIX_RE.fullmatch(value) and any(char.isupper() or char.isdigit() for char in value))


def _candidate_order(item: dict[str, Any], discriminator: str) -> tuple[int, str, int, str]:
    raw_comment_index = item.get("comment_index")
    try:
        comment_index = int(raw_comment_index)
    except (TypeError, ValueError):
        comment_index = 0
    raw_attempt_index = item.get("attempt_index")
    try:
        attempt_index = int(raw_attempt_index)
    except (TypeError, ValueError):
        attempt_index = 0
    tie_breaker = f"{item.get('attachment_id') or item.get('event_id') or ''}:{discriminator}"
    return comment_index, str(item.get("created_at") or ""), attempt_index, tie_breaker


def _candidate_source(item: dict[str, Any], archive_entry: str | None = None) -> dict[str, Any]:
    keys = (
        "attachment_id",
        "event_id",
        "target",
        "filename",
        "canonical_filename",
        "attempt_index",
        "created_at",
        "comment_index",
        "role",
    )
    source = {key: item.get(key) for key in keys if key in item}
    if archive_entry is not None:
        source["archive_entry"] = archive_entry
    return source


def _assignment_matches_homework(homework_id: str, assignment: str) -> bool:
    if not homework_id:
        return True
    try:
        number = int(homework_id)
    except ValueError:
        return True
    text = assignment.casefold()
    if not text:
        return False
    number_pattern = rf"0*{number}(?!\d)"
    return bool(re.search(rf"(?:hw|homework|дз|домашн\w*)\D*{number_pattern}", text))


def _is_gradeable_target(item: dict[str, Any]) -> bool:
    target = Path(str(item.get("target") or ""))
    suffix = Path(str(item.get("filename") or target.name)).suffix.casefold()
    if suffix not in _GRADEABLE_SUFFIXES:
        return False
    if item.get("role") not in (None, "", "student"):
        return False
    return target.is_file()


def _write_prepared_manifest(run_dir: Path, students: list[dict[str, Any]], warnings: list[str]) -> None:
    _atomic_write_json(
        run_dir / "prepared_submissions.json",
        {
            "students": students,
            "warnings": warnings,
            "prepared_at": _now(),
        },
    )


def _prepared_student_id(interaction: dict[str, Any]) -> str:
    student = _safe_display_segment(str(interaction.get("student") or "student"))
    submission = _safe_segment(str(interaction.get("submission_id") or "submission"))
    return f"{student}__{submission}"


def _safe_filename(name: str) -> str:
    filename = Path(name).name
    if not filename or filename in {".", ".."} or "\x00" in filename:
        raise ValueError(f"unsafe attachment filename: {name!r}")
    return filename


def _unsafe_archive_name(name: str) -> bool:
    if not name or "\x00" in name:
        return True
    normalized = name.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return True
    return any(part in {"", ".", ".."} for part in normalized.split("/"))


def _safe_segment(value: str) -> str:
    slug = _SAFE_NAME_RE.sub("_", value.strip()).strip("._-")
    return slug[:120] or "unknown"


def _safe_display_segment(value: str) -> str:
    segment = re.sub(r"[\s/\\:\x00]+", "_", value.strip())
    segment = segment.strip("._-")
    if not segment or segment in {".", ".."}:
        return "unknown"
    return segment[:120]


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    tools_path = str(repo_root / "tools")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{tools_path}{os.pathsep}{current}" if current else tools_path
    return env


def _downloader_script(repo_root: Path) -> Path:
    return repo_root / "dev" / "scripts" / "download_dataschool_submissions.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    payload = {"updated_at": _now(), **state}
    _atomic_write_json(run_dir / "state.json", payload)


def _atomic_write_json(path: Path, data: Any) -> None:
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
            json.dump(data, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
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
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
