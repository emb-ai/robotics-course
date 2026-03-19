"""Integration tests for bot handlers (mocked Telegram and Redis)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.handlers import start_cmd, help_cmd, grades_cmd, grade_cmd, _extract_week


def _make_update(message_text: str | None = None, user_id: int = 1, chat_id: int = 1):
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = message_text
    update.message.reply_text = AsyncMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    return update


def test_start_cmd():
    update = _make_update()
    context = MagicMock()
    asyncio.run(start_cmd(update, context))
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "homework" in call_args.lower() or "grade" in call_args.lower()


def test_help_cmd():
    update = _make_update()
    context = MagicMock()
    asyncio.run(help_cmd(update, context))
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "01" in call_args or "week" in call_args.lower()
    assert "grades" in call_args.lower() or "leaderboard" in call_args.lower()


def test_grades_cmd_empty():
    update = _make_update()
    context = MagicMock()
    with patch("autograder.grades.get_my_grades", return_value=[]):
        asyncio.run(grades_cmd(update, context))
    update.message.reply_text.assert_called_once_with("No grades yet.")


def test_grades_cmd_with_data():
    update = _make_update()
    context = MagicMock()
    with patch("autograder.grades.get_my_grades", return_value=[
        {"week_id": "01", "problem_id": "beads", "passed": 1},
    ]):
        asyncio.run(grades_cmd(update, context))
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "01" in call_args
    assert "beads" in call_args


def test_grade_cmd_valid_week():
    update = _make_update("/grade 01")
    context = MagicMock()
    asyncio.run(grade_cmd(update, context))
    update.message.reply_text.assert_called_once()
    assert "send" in update.message.reply_text.call_args[0][0].lower()


def test_grade_cmd_invalid_week():
    update = _make_update("/grade 99")
    context = MagicMock()
    asyncio.run(grade_cmd(update, context))
    update.message.reply_text.assert_called_once()
    assert "unknown" in update.message.reply_text.call_args[0][0].lower() or "01" in update.message.reply_text.call_args[0][0]
