# Oracle

Orchestration for LLM-based communication with students. The oracle exposes an HTTP API used by the bot (Q&A) and autograder (homework feedback).

## Prerequisites

- An OpenAI-compatible LLM server (vLLM, SGLang, etc.) running and reachable at `ORACLE_LLM_BASE_URL`.

  **Alternatively**, use the bundled Docker image to run both the SGLang LLM and Oracle in one container. The image clones the repo from GitHub (no dev/).

## Tools

The oracle uses lightweight tools to support course Q&A and homework feedback:

### 1. Text search (RAG)

**Purpose:** Find relevant course materials when answering questions.

**How it works:**
- **Python files** (.py): Uses `ripgrep` for fast lexical search over library and homework code.
- **Notebooks** (.ipynb): Uses `nbformat` to extract markdown and code cells, then substring search over class and homework notebooks.

**Flow:** When a student asks a question (and it’s not a short greeting), the oracle runs search, injects snippets into the prompt as `<CONTEXT>`, and the LLM uses them for grounded answers.

### 2. Python code runner

**Purpose:** Run small code snippets students ask about (e.g. “what does this print?”).

**How it works:**
- Writes code to a temporary directory.
- Runs `python main.py` in a subprocess with a configurable timeout (default 10 s, max 30 s).
- Returns stdout, stderr, and exit code. Output is truncated to avoid oversized responses.

**API:** `POST /tools/run_python` with `{"code": "...", "timeout_sec": 10, "stdin": ""}`.

**Security:** The current implementation uses a subprocess with timeout and a private temp dir. For stronger isolation, run student code in a separate container (e.g. via `docker run`) or a dedicated code-runner service with resource limits.

### 3. Homework spec loader

**Purpose:** Load the homework specification for a given week to guide feedback.

**How it works:** Reads `tools/config/weeks.yaml` and per-homework config to resolve the topic slug, then loads `{topic}/homework/homework.ipynb` via nbformat.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/agent/chat` | Q&A: messages + context → assistant text |
| POST | `/agent/feedback` | Homework feedback: code + test results → pedagogic feedback |
| POST | `/tools/run_python` | Execute Python code (timeout, temp dir) |

### `POST /agent/chat`

Request:
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "context": {"chat_id": 123, "user_id": 456, "is_private": true}
}
```

Response:
```json
{"text": "..."}
```

### `POST /agent/feedback`

Request:
```json
{
  "week_id": "01",
  "files": {"so101_ik.py": "..."},
  "exit_code": 1,
  "stdout": "...",
  "stderr": "...",
  "problem_results": {"so101_ik": 0}
}
```

Response:
```json
{"feedback": "..."}
```

### `POST /tools/run_python`

Request:
```json
{
  "code": "print(1 + 1)",
  "timeout_sec": 10,
  "stdin": ""
}
```

Response:
```json
{"exit_code": 0, "stdout": "2\n", "stderr": ""}
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible API base (no trailing slash) |
| `ORACLE_LLM_MODEL` | `glm-5-fp8` | Model name |
| `AI_ROBOTICS_REPO_ROOT` | (inferred) | Repo root for course materials search |

## Running locally

```bash
./tools/run_oracle.sh
```

Listens on `http://0.0.0.0:9000`. The bot and autograder call it when `ORACLE_ENABLED=true` and `ORACLE_BASE_URL` are set.

## Docker (SGLang LLM + Oracle in one container)

Build from the repository root. The image **clones** the repo from GitHub (no local COPY of dev/ or active code):

```bash
docker build -f tools/oracle/Dockerfile -t oracle:latest .
```

For a private repo:

```bash
docker build -f tools/oracle/Dockerfile -t oracle:latest \
  --build-arg REPO_GIT_URL="https://x-access-token:TOKEN@github.com/org/repo.git" .
```

Run (requires NVIDIA GPU and nvidia-docker):

```bash
docker run --gpus all --ipc=host -p 9000:9000 -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  oracle:latest
```

- **SGLang** serves the model on port 8000.
- **Oracle** listens on port 9000.

### Docker environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_MODEL` | `Qwen/Qwen2-0.5B-Instruct` | Model for SGLang |
| `SGLANG_PORT` | `8000` | Port for SGLang |
| `ORACLE_PORT` | `9000` | Port for Oracle |
| `SGLANG_EXTRA_ARGS` | (empty) | Extra args for `sglang.launch_server` |

For production (e.g. GLM-5 on 8× H100):

```bash
docker run --gpus all --ipc=host -p 9000:9000 -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e SGLANG_MODEL=zai-org/GLM-5-FP8 \
  -e SGLANG_EXTRA_ARGS="--tp 8" \
  oracle:latest
```

### cloud.ru Distributed Train (`Dockerfile.cloudru`)

Image bakes **`tools/`** at `/opt/oracle/tools` and clones **[emb-ai/robotics-course](https://github.com/emb-ai/robotics-course)** branch **`2026`** to `/opt/course-repo`. `AI_ROBOTICS_REPO_ROOT` points at that tree (homework notebooks). On each job start, `launch_oracle.sh` runs `git pull --ff-only` when the job has internet. Build/push: `tools/deploy/push-oracle-cloudru.sh`; job script: **`/opt/oracle/launch_oracle.sh`**. Override **`COURSE_REPO_URL`** / **`COURSE_REPO_BRANCH`** via job env if needed.
