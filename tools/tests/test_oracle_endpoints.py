"""Integration tests for Oracle FastAPI endpoints."""

from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from oracle.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_agent_chat_empty_messages(client):
    r = client.post("/agent/chat", json={"messages": [], "context": {"chat_id": 1, "user_id": 1, "is_private": True}})
    assert r.status_code == 200
    assert "No message" in r.json()["text"]


def test_agent_chat_mocked_llm(client):
    with patch("oracle.llm_client._get_client") as m:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"choices": [{"message": {"content": "Hello from TA!"}}]})
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        m.return_value = mock_client

        r = client.post(
            "/agent/chat",
            json={
                "messages": [{"role": "user", "content": "What is kinematics?"}],
                "context": {"chat_id": 1, "user_id": 1, "is_private": True},
            },
        )
        assert r.status_code == 200
        assert "Hello from TA!" in r.json()["text"]


def test_agent_feedback_unknown_week(client):
    r = client.post(
        "/agent/feedback",
        json={
            "week_id": "99",
            "files": {"a.py": "x=1"},
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "problem_results": {},
        },
    )
    assert r.status_code == 200
    assert "Unknown" in r.json()["feedback"] or "99" in r.json()["feedback"]


def test_agent_feedback_mocked_llm(client):
    with patch("oracle.llm_client._get_client") as m:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"choices": [{"message": {"content": "Check your loop bounds."}}]})
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        m.return_value = mock_client

        r = client.post(
            "/agent/feedback",
            json={
                "week_id": "01",
                "files": {"beads.py": "x=1"},
                "exit_code": 1,
                "stdout": "FAILED",
                "stderr": "",
                "problem_results": {"beads": 0},
            },
        )
        assert r.status_code == 200
        assert "loop" in r.json()["feedback"].lower() or "Check" in r.json()["feedback"]


def test_tools_run_python(client):
    r = client.post("/tools/run_python", json={"code": "print(42)", "timeout_sec": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["exit_code"] == 0
    assert "42" in data["stdout"]
