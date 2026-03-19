"""OpenAI-compatible LLM client (httpx) with agentic tool-calling loop."""

import json
import logging
from typing import Any

import httpx

from shared.week_config import get_repo_root

from . import config as oracle_config
from .prompts import SYSTEM_FEEDBACK, SYSTEM_TA_GROUP, SYSTEM_TA_PERSONA
from .tools.code_runner import run_python
from .tools.file_tools import list_directory, read_file, search_code

logger = logging.getLogger(__name__)

# Shared client for connection reuse (pooling)
_http_client: httpx.AsyncClient | None = None

MAX_TOOL_ITERATIONS = 5

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern in the course codebase. Returns grouped, compact matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern or keyword"},
                    "path": {"type": "string", "description": "Path to search within", "default": "."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents. Use aggressive mode for code to get signatures only (less tokens).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file (relative to repo root)"},
                    "level": {
                        "type": "string",
                        "enum": ["normal", "aggressive"],
                        "description": "normal=full content, aggressive=signatures only for code",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List directory contents in compact form.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (relative to repo root)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"},
                    "timeout_sec": {"type": "integer", "description": "Timeout seconds (max 30)", "default": 10},
                    "stdin": {"type": "string", "description": "Stdin input", "default": ""},
                },
                "required": ["code"],
            },
        },
    },
]


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=oracle_config.ORACLE_TIMEOUT_SEC)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name and return result string."""
    repo_root = get_repo_root()
    try:
        if name == "search_code":
            return search_code(
                args["query"],
                args.get("path", "."),
                base=repo_root,
                max_chars=2000,
            )
        if name == "read_file":
            return read_file(
                args["path"],
                level=args.get("level", "normal"),
                base=repo_root,
            )
        if name == "list_directory":
            return list_directory(args["path"], base=repo_root)
        if name == "run_python":
            code = args["code"]
            timeout = min(int(args.get("timeout_sec", 10)), 30)
            stdin = args.get("stdin", "")
            exit_code, stdout, stderr = run_python(code, timeout, stdin)
            return f"exit_code={exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    except Exception as e:
        return f"Error: {e}"
    return f"Unknown tool: {name}"


def _build_chat_messages(
    messages: list[dict[str, str]],
    context: dict[str, Any],
    extra_context: str = "",
) -> list[dict[str, Any]]:
    is_private = context.get("is_private", True)
    system = SYSTEM_TA_PERSONA if is_private else SYSTEM_TA_GROUP
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.get("role") == "user" and extra_context:
            out.append({"role": "user", "content": f"<CONTEXT>\n{extra_context}\n</CONTEXT>\n\n{m['content']}"})
            extra_context = ""
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


async def _chat_with_tools(
    messages: list[dict[str, Any]],
    use_tools: bool = True,
) -> str:
    """Call LLM with optional tool-calling loop."""
    url = f"{oracle_config.ORACLE_LLM_BASE_URL.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": oracle_config.ORACLE_LLM_MODEL,
        "messages": messages,
        "max_tokens": 1024,
    }
    if use_tools:
        payload["tools"] = TOOL_DEFINITIONS
        payload["tool_choice"] = "auto"

    client = _get_client()
    for _ in range(MAX_TOOL_ITERATIONS):
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            # Append assistant message with tool_calls
            messages.append(msg)
            for tc in tool_calls:
                tid = tc.get("id", "")
                fn = tc.get("function", {})
                fname = fn.get("name", "")
                try:
                    fargs = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    fargs = {}
                result = _execute_tool(fname, fargs)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result[:8000],
                })
        else:
            return content.strip()
    return "Max tool iterations reached. Please try a simpler question."


async def chat_completion_async(
    messages: list[dict[str, str]],
    context: dict[str, Any],
    extra_context: str = "",
) -> str:
    """Call LLM for chat with agentic tool-calling. Returns assistant text."""
    built = _build_chat_messages(messages, context, extra_context)
    return await _chat_with_tools(built, use_tools=True)


async def feedback_completion(
    week_id: str,
    files: dict[str, str],
    exit_code: int,
    stdout: str,
    stderr: str,
    problem_results: dict[str, int],
    homework_spec: str,
) -> str:
    """Generate pedagogic feedback for a homework submission, with optional tool use."""
    code_block = "\n\n---\n\n".join(
        f"# {name}\n{content}" for name, content in files.items()
    )
    results_str = ", ".join(f"{k}: {'pass' if v else 'fail'}" for k, v in problem_results.items())
    user_content = f"""Homework week {week_id}. Test results: {results_str} (exit code {exit_code}).

=== Student code ===
{code_block}

=== stdout ===
{stdout}

=== stderr ===
{stderr}
"""
    if homework_spec:
        user_content = f"<HOMEWORK_SPEC>\n{homework_spec[:4000]}\n</HOMEWORK_SPEC>\n\n" + user_content

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_FEEDBACK},
        {"role": "user", "content": user_content},
    ]
    return await _chat_with_tools(messages, use_tools=True)
