"""Persist full pytest/Docker stdout+stderr for each grading submission."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from shared.schemas import Job


def get_grading_logs_root() -> Path:
    """Directory root for per-submission logs (default: tools/data/grading_logs)."""
    base = os.environ.get("GRADING_LOGS_PATH")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data" / "grading_logs"


def save_grading_log(
    job: Job,
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    elapsed_sec: float | None = None,
    status: str = "completed",
    error: str | None = None,
) -> Path:
    """
    Write a UTF-8 text log for one submission. Returns path to the file.

    status: completed | timeout | docker_error | unknown_week
    """
    root = get_grading_logs_root()
    week_dir = root / job.week_id
    week_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]  # ms precision
    safe_name = f"{ts}_user{job.user_id}.log"
    path = week_dir / safe_name

    files_list = ", ".join(sorted(job.files.keys())) if job.files else "(none)"
    header = (
        f"week_id: {job.week_id}\n"
        f"user_id: {job.user_id}\n"
        f"chat_id: {job.chat_id}\n"
        f"first_name: {job.first_name or ''}\n"
        f"username: {job.username or ''}\n"
        f"status: {status}\n"
        f"exit_code: {exit_code if exit_code is not None else 'n/a'}\n"
        f"elapsed_sec: {elapsed_sec if elapsed_sec is not None else 'n/a'}\n"
        f"submitted_files: {files_list}\n"
    )
    if error:
        header += f"error: {error}\n"
    body = (
        "\n=== stdout ===\n"
        + (stdout or "")
        + "\n\n=== stderr ===\n"
        + (stderr or "")
        + "\n"
    )
    path.write_text(header + body, encoding="utf-8", errors="replace")
    return path
