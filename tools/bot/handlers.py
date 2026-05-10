"""Telegram bot handlers."""

import io
import logging
import zipfile
from pathlib import Path

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from autograder.week_registry import get_metrics_config, get_points, get_solution_files, list_weeks
from shared.schemas import Job
from shared.week_helpers import WEEK_PATTERN, extract_week as _extract_week

from . import config as bot_config
from .queue_client import push_job
from allowlist_db.store import is_allowed

logger = logging.getLogger(__name__)


def _check_rate_limit(user_id: int, week_id: str) -> tuple[bool, int]:
    """Return (allowed, seconds_remaining). If rate limit disabled, (True, 0)."""
    if bot_config.RATE_LIMIT_SEC <= 0:
        return True, 0
    from shared.redis_pool import get_redis

    r = get_redis()
    key = f"{bot_config.RATE_LIMIT_KEY_PREFIX}{user_id}:{week_id}"
    ttl = r.ttl(key)
    if ttl > 0:
        return False, ttl
    r.setex(key, bot_config.RATE_LIMIT_SEC, "1")
    return True, 0


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the AI in Robotics homework grader. "
        "Send your solution files with caption /grade 01 (or week 02, etc.). "
        "Commands: /grades, /leaderboard, /help"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    weeks = ", ".join(list_weeks())
    await update.message.reply_text(
        f"<b>How to submit</b>\n"
        f"Send .py documents or a .zip with caption <code>/grade 01</code> (or week 02, etc.).\n\n"
        f"<b>Available weeks</b>: {weeks}\n\n"
        f"<b>Commands</b>\n"
        f"/grades — your scores\n"
        f"/leaderboard — all students",
        parse_mode="HTML",
    )


async def grades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from autograder.grades import get_my_grades

    user_id = update.effective_user.id
    rows = get_my_grades(user_id)
    if not rows:
        await update.message.reply_text("No grades yet.")
        return
    lines = ["Week | Problem | Pts"]
    for r in rows:
        pts = r.get("points")
        max_pts = None
        try:
            max_pts = get_points(r["week_id"]).get(r["problem_id"], 1)
        except ValueError:
            pass
        if pts is not None and max_pts is not None:
            lines.append(f"{r['week_id']} | {r['problem_id']} | {pts}/{max_pts}")
        else:
            status = "✓" if r["passed"] else "✗"
            lines.append(f"{r['week_id']} | {r['problem_id']} | {status}")
    await update.message.reply_text("<pre>" + "\n".join(lines) + "</pre>", parse_mode="HTML")


async def leaderboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from autograder.grades.store import get_leaderboard_pivoted, get_metric_leaderboard

    parts = []
    # Points leaderboard
    rows = get_leaderboard_pivoted()
    if rows:
        cols = ["display", "total_pts"]
        for r in rows:
            for k in r:
                if k not in ("telegram_id", "first_name", "username", "display", "total_pts") and k not in cols:
                    cols.append(k)
        cols = ["display", "total_pts"] + sorted([c for c in cols if c not in ("display", "total_pts")])
        lines = ["\t".join(cols)]
        for r in rows:
            line = [str(r.get(c, "")) for c in cols]
            lines.append("\t".join(line))
        parts.append("<b>Points</b>\n<pre>" + "\n".join(lines) + "</pre>")

    # Metric leaderboards (per week/problem with metrics config)
    for week_id in list_weeks():
        try:
            metrics_cfg = get_metrics_config(week_id)
        except ValueError:
            continue
        for problem_id, cfg in metrics_cfg.items():
            name = cfg.get("name", problem_id)
            direction = cfg.get("direction", "minimize")
            m_rows = get_metric_leaderboard(week_id, problem_id, direction, limit=10)
            if m_rows:
                lines = [f"{r['display']}: {r['metric_value']}" for r in m_rows]
                parts.append(f"<b>Week {week_id} — {name}</b>\n<pre>" + "\n".join(lines) + "</pre>")

    if not parts:
        await update.message.reply_text("No grades yet.")
        return
    text = "\n\n".join(parts)
    if len(text) > 4000:
        text = text[:3900] + "\n...(truncated, see dashboard)"
    await update.message.reply_text(text, parse_mode="HTML")


async def grade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /grade 01 - ask user to send files."""
    text = (update.message.text or "").strip()
    week = _extract_week(text)
    if not week or week not in list_weeks():
        await update.message.reply_text(f"Usage: /grade 01. Available weeks: {', '.join(list_weeks())}")
        return
    await update.message.reply_text(f"Now send your solution files (.py or .zip) for week {week}.")


async def _collect_files_from_docs(update: Update) -> dict[str, str] | str:
    """
    Collect .py files from document(s). Returns dict filename->content or error str.
    Handles: single .py, multiple .py, single .zip
    """
    files: dict[str, str] = {}
    docs = []
    if update.message.document:
        docs.append(update.message.document)
    if update.message.photo:
        return "Send documents (.py or .zip), not photos."

    for doc in docs:
        fname = (doc.file_name or "").lower()
        if fname.endswith(".zip"):
            bio = io.BytesIO()
            file = await doc.get_file()
            await file.download_to_memory(bio)
            bio.seek(0)
            try:
                with zipfile.ZipFile(bio) as zf:
                    for name in zf.namelist():
                        base = Path(name).name if "/" in name or "\\" in name else name
                        if base.endswith(".py"):
                            content = zf.read(name).decode("utf-8", errors="replace")
                            files[base] = content
            except zipfile.BadZipFile:
                return "Invalid zip file."
        elif fname.endswith(".py"):
            base = Path(doc.file_name or "file.py").name
            bio = io.BytesIO()
            file = await doc.get_file()
            await file.download_to_memory(bio)
            content = bio.getvalue().decode("utf-8", errors="replace")
            files[base] = content
        else:
            return f"Unknown file type: {doc.file_name}. Send .py or .zip only."

    if not files:
        return "No valid .py files found."
    return files


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return

    user = update.effective_user
    if not user:
        return

    if not is_allowed(user.id):
        await update.message.reply_text("You are not allowed to use this bot.")
        return

    # Get week from caption
    caption = update.message.caption or ""
    week = _extract_week(caption)
    if not week:
        await update.message.reply_text(
            "Add the week to your message, e.g. <code>/grade 01</code> or <code>week 01</code> in the caption.",
            parse_mode="HTML",
        )
        return

    if week not in list_weeks():
        await update.message.reply_text(f"Unknown week. Available: {', '.join(list_weeks())}")
        return

    allowed_files = set(get_solution_files(week))
    result = await _collect_files_from_docs(update)
    if isinstance(result, str):
        await update.message.reply_text(result)
        return

    files = result
    # Filter to allowed names only (partial submission OK)
    filtered = {k: v for k, v in files.items() if k in allowed_files}
    unknown = set(files) - allowed_files
    if unknown:
        await update.message.reply_text(
            f"Ignoring unknown files: {', '.join(sorted(unknown))}. "
            f"Expected for week {week}: {', '.join(sorted(allowed_files))}."
        )
    if not filtered:
        await update.message.reply_text(
            f"No valid solution files. Expected: {', '.join(sorted(allowed_files))}."
        )
        return

    allowed, secs = _check_rate_limit(user.id, week)
    if not allowed:
        await update.message.reply_text(
            f"Rate limit: wait {secs} seconds before resubmitting week {week}."
        )
        return

    job = Job(
        chat_id=update.effective_chat.id,
        week_id=week,
        files=filtered,
        user_id=user.id,
        first_name=user.first_name,
        username=user.username,
    )
    try:
        push_job(job)
    except Exception as e:
        logger.exception("Failed to enqueue job for user=%s week=%s", user.id, week)
        await update.message.reply_text(
            "Failed to queue your submission (queue unavailable). Please try again in a few minutes."
        )
        return
    await update.message.reply_text(f"Queued for week {week}. You'll get results here shortly.")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: route to oracle for Q&A when enabled."""
    if not update.message or not update.message.text:
        return
    user = update.effective_user
    if not user:
        return
    is_private = update.effective_chat.type == "private"
    if is_private and not is_allowed(user.id):
        await update.message.reply_text("You are not allowed to use this bot.")
        return
    if not bot_config.ORACLE_ENABLED or not bot_config.ORACLE_BASE_URL:
        await update.message.reply_text("Ask me about the course or send /grade NN to submit homework.")
        return
    text = update.message.text.strip()
    url = f"{bot_config.ORACLE_BASE_URL.rstrip('/')}/agent/chat"
    payload = {
        "messages": [{"role": "user", "content": text}],
        "context": {"chat_id": update.effective_chat.id, "user_id": user.id, "is_private": is_private},
    }
    try:
        async with httpx.AsyncClient(timeout=bot_config.ORACLE_TIMEOUT_SEC) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("Oracle request failed: %s", e)
        await update.message.reply_text("Oracle unavailable, try later.")
        return
    response_text = data.get("text", "")
    if len(response_text) > 4096:
        response_text = response_text[:4093] + "..."
    await update.message.reply_text(response_text)
