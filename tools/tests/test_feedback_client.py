"""Tests for the OpenAI-compatible feedback LLM client."""

import httpx

from autograder.feedback.client import FeedbackClientConfig, generate_feedback
from autograder.feedback.prompt_builder import FeedbackPrompt


def _prompt():
    return FeedbackPrompt(
        system_prompt="system feedback prompt",
        user_prompt="user feedback prompt",
        metadata={"prompt_token_budget_approx": 12},
    )


def test_generate_feedback_skips_when_endpoint_or_model_missing(monkeypatch):
    monkeypatch.delenv("ORACLE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ORACLE_LLM_MODEL", raising=False)

    result = generate_feedback(_prompt())

    assert result.status == "skipped"
    assert "ORACLE_LLM_BASE_URL" in result.error


def test_generate_feedback_posts_openai_compatible_payload_with_bearer_header():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Check the Jacobian dimensions."}}]},
        )

    result = generate_feedback(
        _prompt(),
        FeedbackClientConfig(
            base_url="https://example.test/v1",
            model="demo-model",
            api_key="secret-key",
            timeout_sec=5,
            transport=httpx.MockTransport(handler),
        ),
    )

    assert result.status == "ok"
    assert result.content == "Check the Jacobian dimensions."
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer secret-key"
    assert '"model":"demo-model"' in captured["payload"]
    assert '"max_tokens":256' in captured["payload"]
    assert "system feedback prompt" in captured["payload"]
    assert "user feedback prompt" in captured["payload"]


def test_generate_feedback_omits_authorization_header_without_api_key():
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Draft"}}]})

    result = generate_feedback(
        _prompt(),
        FeedbackClientConfig(
            base_url="https://example.test/v1",
            model="demo-model",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert result.status == "ok"
    assert "authorization" not in captured["headers"]


def test_generate_feedback_reports_http_error():
    result = generate_feedback(
        _prompt(),
        FeedbackClientConfig(
            base_url="https://example.test/v1",
            model="demo-model",
            transport=httpx.MockTransport(lambda request: httpx.Response(500, text="nope")),
        ),
    )

    assert result.status == "error"
    assert "500" in result.error


def test_generate_feedback_reports_timeout():
    def handler(request):
        raise httpx.TimeoutException("slow", request=request)

    result = generate_feedback(
        _prompt(),
        FeedbackClientConfig(
            base_url="https://example.test/v1",
            model="demo-model",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert result.status == "timeout"
    assert "slow" in result.error
