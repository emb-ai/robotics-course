"""Synchronous OpenAI-compatible client for feedback draft generation."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Literal

import httpx

from .prompt_builder import FeedbackPrompt


FeedbackStatus = Literal["ok", "skipped", "error", "timeout"]


@dataclass(frozen=True)
class FeedbackClientConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_sec: float = 60.0
    max_tokens: int = 0
    transport: httpx.BaseTransport | None = dataclasses.field(default=None, compare=False)

    @classmethod
    def from_env(cls) -> "FeedbackClientConfig":
        return cls(
            base_url=os.environ.get("ORACLE_LLM_BASE_URL", "").strip(),
            model=os.environ.get("ORACLE_LLM_MODEL", "").strip(),
            api_key=os.environ.get("ORACLE_LLM_API_KEY", "").strip(),
            timeout_sec=_env_timeout(),
            max_tokens=_env_max_tokens(),
        )


@dataclass(frozen=True)
class FeedbackResult:
    status: FeedbackStatus
    content: str = ""
    error: str = ""
    model: str = ""


def generate_feedback(
    prompt: FeedbackPrompt,
    config: FeedbackClientConfig | None = None,
) -> FeedbackResult:
    """Generate one feedback draft, returning structured fail-soft status."""

    active_config = config or FeedbackClientConfig.from_env()
    if not active_config.base_url or not active_config.model:
        return FeedbackResult(
            status="skipped",
            error="Missing ORACLE_LLM_BASE_URL or ORACLE_LLM_MODEL.",
            model=active_config.model,
        )

    url = f"{active_config.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if active_config.api_key:
        headers["Authorization"] = f"Bearer {active_config.api_key}"
    payload = {
        "model": active_config.model,
        "messages": prompt.messages,
        "max_tokens": active_config.max_tokens or _env_max_tokens(),
    }
    payload.update(_model_payload_options(active_config.model))

    try:
        with httpx.Client(
            timeout=active_config.timeout_sec,
            transport=active_config.transport,
        ) as client:
            response = client.post(
                url,
                headers=headers,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        return FeedbackResult(status="timeout", error=str(exc), model=active_config.model)
    except httpx.HTTPStatusError as exc:
        return FeedbackResult(status="error", error=str(exc), model=active_config.model)
    except httpx.RequestError as exc:
        return FeedbackResult(status="error", error=str(exc), model=active_config.model)
    except (ValueError, TypeError) as exc:
        return FeedbackResult(status="error", error=f"Invalid LLM response: {exc}", model=active_config.model)

    content = _extract_content(data)
    if not content:
        return FeedbackResult(status="error", error="LLM response did not include message content.", model=active_config.model)
    return FeedbackResult(status="ok", content=content, model=active_config.model)


def _extract_content(data: dict) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    first = choices[0] if isinstance(choices, list) else {}
    text = first.get("text") if isinstance(first, dict) else ""
    if text:
        return str(text).strip()
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {None, "text", "output_text"}:
                parts.append(str(item.get("text") or item.get("content") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _model_payload_options(model: str) -> dict:
    if "qwen" not in model.casefold() or _env_truthy("ORACLE_LLM_ENABLE_THINKING"):
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _env_timeout() -> float:
    raw = os.environ.get("ORACLE_TIMEOUT_SEC", "").strip()
    if not raw:
        return 60.0
    try:
        return float(raw)
    except ValueError:
        return 60.0


def _env_max_tokens() -> int:
    raw = os.environ.get("ORACLE_LLM_MAX_TOKENS", "").strip()
    if not raw:
        return 1024
    try:
        value = int(raw)
    except ValueError:
        return 1024
    return max(128, min(value, 8192))
