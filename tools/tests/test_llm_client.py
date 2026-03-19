"""Tests for oracle/llm_client.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oracle.llm_client import _build_chat_messages, chat_completion_async, feedback_completion


def test_build_chat_messages_private():
    messages = [{"role": "user", "content": "Hi"}]
    ctx = {"is_private": True}
    built = _build_chat_messages(messages, ctx)
    assert built[0]["role"] == "system"
    assert "YSDA" in built[0]["content"]


def test_build_chat_messages_with_extra_context():
    messages = [{"role": "user", "content": "Explain kinematics"}]
    ctx = {"is_private": True}
    built = _build_chat_messages(messages, ctx, extra_context="[snippet from repo]")
    assert len(built) >= 2
    user_msg = built[-1]
    assert "CONTEXT" in user_msg["content"]
    assert "kinematics" in user_msg["content"]


def test_chat_completion_async_mocked():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"choices": [{"message": {"content": "Hello!"}}]})
    with patch("oracle.llm_client._get_client") as m:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        m.return_value = mock_client
        text = asyncio.run(chat_completion_async(
            [{"role": "user", "content": "Hi"}],
            {"is_private": True},
        ))
        assert text == "Hello!"


def test_feedback_completion_mocked():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"choices": [{"message": {"content": "Try checking the formula."}}]})
    with patch("oracle.llm_client._get_client") as m:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        m.return_value = mock_client
        text = asyncio.run(feedback_completion(
            week_id="01",
            files={"beads.py": "x=1"},
            exit_code=1,
            stdout="FAILED",
            stderr="",
            problem_results={"beads": 0},
            homework_spec="",
        ))
        assert "formula" in text.lower() or "Try" in text
