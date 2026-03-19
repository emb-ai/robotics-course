#!/usr/bin/env python3
"""Oracle FastAPI app: /agent/chat, /agent/feedback, /health."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from pydantic import BaseModel

from . import config as oracle_config
from .llm_client import chat_completion_async, feedback_completion, close_http_client
from .materials import get_topic_slug, load_homework_spec_text
from .tools.code_runner import run_python

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_http_client()


app = FastAPI(title="Oracle", description="LLM orchestration for course TA", lifespan=lifespan)


class ChatContext(BaseModel):
    chat_id: int
    user_id: int
    is_private: bool = True
    week_id: str | None = None


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    context: ChatContext


class ChatResponse(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    week_id: str
    files: dict[str, str]
    exit_code: int
    stdout: str
    stderr: str
    problem_results: dict[str, int]


class FeedbackResponse(BaseModel):
    feedback: str


class RunPythonRequest(BaseModel):
    code: str
    timeout_sec: int = 10
    stdin: str = ""


class RunPythonResponse(BaseModel):
    exit_code: int
    stdout: str
    stderr: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """Q&A: agentic tool-calling (search, read_file, list_dir, run_python)."""
    if not req.messages:
        return ChatResponse(text="No message received.")
    last_msg = req.messages[-1]
    if last_msg.get("role") != "user":
        return ChatResponse(text="Please send a user message.")
    ctx = {
        "chat_id": req.context.chat_id,
        "user_id": req.context.user_id,
        "is_private": req.context.is_private,
    }
    text = await chat_completion_async(req.messages, ctx, extra_context="")
    return ChatResponse(text=text)


@app.post("/agent/feedback", response_model=FeedbackResponse)
async def agent_feedback(req: FeedbackRequest):
    """Generate pedagogic feedback for a homework submission."""
    try:
        topic_slug = get_topic_slug(req.week_id)
    except ValueError as e:
        return FeedbackResponse(feedback=f"Unknown week: {e}")
    homework_spec = load_homework_spec_text(req.week_id)
    feedback = await feedback_completion(
        week_id=req.week_id,
        files=req.files,
        exit_code=req.exit_code,
        stdout=req.stdout,
        stderr=req.stderr,
        problem_results=req.problem_results,
        homework_spec=homework_spec,
    )
    return FeedbackResponse(feedback=feedback)


@app.post("/tools/run_python", response_model=RunPythonResponse)
async def tools_run_python(req: RunPythonRequest):
    """Execute Python code in a restricted subprocess (timeout, temp dir)."""
    exit_code, stdout, stderr = run_python(
        code=req.code,
        timeout_sec=min(req.timeout_sec, 30),  # Cap at 30s
        stdin=req.stdin,
    )
    return RunPythonResponse(exit_code=exit_code, stdout=stdout, stderr=stderr)
