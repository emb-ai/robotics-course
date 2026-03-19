"""Process one grading job: run Docker, parse output, store grades, send Telegram."""

import html
import json
import logging
import subprocess

import requests
from shared.autograder_telemetry import mark_job_finished, mark_job_started
from shared.schemas import Job

from . import config as autograder_config
from .docker_runner import run as docker_run
from .pytest_parser import parse_metrics, parse_pytest_output
from .week_registry import get_metrics_config, get_points, get_problem_ids

logger = logging.getLogger(__name__)

def _send_telegram(chat_id: int, text: str, document: bytes | None = None) -> None:
    import os

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return
    base = f"https://api.telegram.org/bot{token}"

    import requests

    if document:
        # Caption limit 1024 chars
        r = requests.post(
            f"{base}/sendDocument",
            data={"chat_id": chat_id, "caption": text[:1024] if text else "", "parse_mode": "HTML"},
            files={"document": ("result.txt", document, "text/plain")},
        )
    else:
        r = requests.post(
            f"{base}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        )
    r.raise_for_status()


def format_result(
    exit_code: int,
    stdout: str,
    stderr: str,
    max_inline: int = 3500,
) -> tuple[str, bytes | None]:
    """
    Format grading result for Telegram.
    Returns (message_text, optionally document_bytes if full output attached).
    """
    full = f"=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}"
    full_stripped = full.strip()

    # Summary line
    if exit_code == 0:
        summary = "✅ All tests passed."
    else:
        summary = "❌ Some tests failed."

    # First failure snippet (first FAILED block)
    snippet = ""
    for line in full_stripped.split("\n"):
        if "FAILED" in line or "Error" in line.lower():
            snippet = line[:500]
            break
    if snippet:
        summary += f"\n\nFirst failure:\n<code>{html.escape(snippet[:500])}</code>"

    if len(full_stripped) <= max_inline:
        return summary + "\n\n<pre>" + html.escape(full_stripped[:3000]) + "</pre>", None

    return summary + "\n\n(Full output attached.)", full_stripped.encode("utf-8")


def process_job(job: Job) -> None:
    """Run grading for one job; store grades; send result to user."""
    try:
        problem_ids = get_problem_ids(job.week_id)
    except ValueError as e:
        _send_telegram(job.chat_id, f"Error: {e}")
        return

    mark_job_started(job)
    try:
        exit_code, stdout, stderr = docker_run(job.week_id, job.files)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Job timed out week=%s chat=%s (docker compose exceeded time limit; "
            "first image build/pull can take many minutes)",
            job.week_id,
            job.chat_id,
        )
        mark_job_finished(
            job,
            "timeout",
            "Docker compose subprocess limit reached (build/pull + tests). "
            "Set higher limits in homework autograder.yaml or AUTOGRADER_DOCKER_OVERHEAD_SEC.",
        )
        _send_telegram(
            job.chat_id,
            "⏱ Run timed out. If this was your first submission, the grader may still be "
            "building the homework Docker image (can take several minutes). Try again in a few minutes.",
        )
        return
    except Exception as e:
        logger.exception("Docker run failed")
        mark_job_finished(job, "docker_error", str(e))
        _send_telegram(job.chat_id, f"Error running grader: {e}")
        return

    # Parse and store grades
    problem_results = parse_pytest_output(stdout, stderr, problem_ids)
    metrics_raw = parse_metrics(stdout, stderr)
    metrics_cfg = get_metrics_config(job.week_id)

    if problem_results or metrics_raw:
        from .grades.schema import get_connection
        from .grades.store import upsert_grades_batch, upsert_metric

        problem_points = get_points(job.week_id)
        conn = get_connection()
        try:
            if problem_results:
                upsert_grades_batch(
                    conn,
                    job.user_id,
                    job.week_id,
                    problem_results,
                    problem_points=problem_points or None,
                    first_name=job.first_name,
                    username=job.username,
                )
            for problem_id, value in metrics_raw.items():
                cfg = metrics_cfg.get(problem_id, {})
                direction = cfg.get("direction", "minimize")
                upsert_metric(conn, job.user_id, job.week_id, problem_id, value, direction)
            conn.commit()
        finally:
            conn.close()

    # Format result
    msg, doc = format_result(exit_code, stdout, stderr)
    summary_line = "✅ All tests passed." if exit_code == 0 else "❌ Some tests failed."

    # Optionally get oracle feedback
    if autograder_config.ORACLE_ENABLED and autograder_config.ORACLE_BASE_URL:
        url = f"{autograder_config.ORACLE_BASE_URL.rstrip('/')}/agent/feedback"
        payload = {
            "week_id": job.week_id,
            "files": job.files,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "problem_results": problem_results,
        }
        try:
            r = requests.post(url, json=payload, timeout=autograder_config.ORACLE_TIMEOUT_SEC)
            r.raise_for_status()
            data = r.json()
            feedback = data.get("feedback", "").strip()
            if feedback:
                msg = f"{summary_line}\n\n{feedback}"
                if len(msg) > 3500:
                    doc = msg.encode("utf-8")
                    msg = summary_line + "\n\n(Full feedback attached.)"
        except Exception as e:
            logger.warning("Oracle feedback request failed: %s", e)

    mark_job_finished(job, "completed", f"exit_code={exit_code}")
    _send_telegram(job.chat_id, msg, doc)


