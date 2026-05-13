# Tools: Bot, Autograder, Oracle

Three separate tools:

- **Bot** (`tools/bot`) — Telegram interface; accepts homework submissions (and will grow beyond that).
- **Autograder** (`tools/autograder`) — Queue daemon, Docker runs, test execution, grade storage, pytest parsing. All grading logic lives here.
- **Oracle** (`tools/oracle`) — LLM orchestration for Q&A and homework feedback (HTTP API).

## Setup

1. **Dependencies** (use conda env `ai_in_robotics`):
   ```bash
   conda activate ai_in_robotics
   pip install -r tools/requirements.txt
   ```

2. **Environment**: Copy `tools/config/.env.example` to `.env` in repo root:
   ```bash
   cp tools/config/.env.example .env
   ```
   Edit `.env` and set `TELEGRAM_BOT_TOKEN`, `REDIS_URL`, etc. For oracle: `ORACLE_ENABLED=true`, `ORACLE_BASE_URL`, `ORACLE_LLM_BASE_URL` (LLM server must be running).

3. **Redis**: Run Redis locally or use a cloud instance. Default: `redis://localhost:6379/0`.

4. **Dev mount** (for full grading with hidden tests): Ensure `dev/<topic>/homework/reference_solution` and `dev/<topic>/homework/hidden_tests` exist. The Docker compose mounts these when present.

5. **Reference vs container**: After homework changes, run `cd tools && pytest tests/test_homework_reference_container.py -v` (container runs **reference** as `solutions/`; autograder passes `GRADING_STUDENT_SUBMISSION=1` so student code is used instead). Docker required; rebuild images if you change Dockerfiles/entrypoints.

6. **Docker**: Compose bind mounts use absolute `REPO_ROOT` (set automatically by each `homework/container/run.sh` and by the autograder). To build manually from repo root:
   ```bash
   export REPO_ROOT="$(pwd -P)"
   docker compose -f 01-intro-and-kinematics/homework/container/docker_compose.yaml build
   ```

## Running

- **Bot**:
  ```bash
  ./tools/run_bot.sh
  ```

- **Autograder daemon**:
  ```bash
  ./tools/run_autograder.sh
  ```

- **Oracle** (optional; for Q&A and feedback, requires LLM server):
  ```bash
  ./tools/run_oracle.sh
  ```
  Or use Docker to run both SGLang LLM and Oracle in one container: `docker build -f tools/oracle/Dockerfile .` (see `tools/oracle/README.md`). The image clones the repo from GitHub (no dev/).

- **Admin dashboards** (bot, autograder with grades):
  ```bash
  ./tools/run_dashboards.sh
  ```
  Then open http://127.0.0.1:5001 (bot), :5002 (autograder: queue, logs, grades table, CSV export). For remote access, use SSH port forwarding:
  ```bash
  ssh -L 5001:127.0.0.1:5001 -L 5002:127.0.0.1:5002 user@host
  ```

## Batch grading cockpit

Batch grading is reports-only staff tooling. It grades one homework against local one-folder-per-student submissions, writes reports under `dev/grading_batches/<run_id>/`, and does not write `tools/data/grades.db` or send Telegram messages.

The batch dashboard is part of the autograder dashboard:

```bash
conda activate ai_in_robotics
export PYTHONPATH="$PWD/tools:$PYTHONPATH"
export ORACLE_LLM_BASE_URL="https://entrypoint/v1"
export ORACLE_LLM_MODEL="Qwen/Qwen3.5-122B-A10B"
export ORACLE_LLM_API_KEY="<local token>"
export ORACLE_TIMEOUT_SEC=60
export ORACLE_LLM_MAX_TOKENS=1024
rtk ./tools/run_dashboards.sh
```

Open http://127.0.0.1:5002/batches. Redis is not required for these batch pages; Redis is only needed for the live queue worker path.

For early smoke tests, prefer the CLI so you can cap DataSchool downloads and worker count. HW1, HW2, and HW3 are all valid batch homework ids; HW3 is first-class even though it is not registered in `tools/config/weeks.yaml` for Telegram grading.

```bash
rtk docker compose -f 03-control/homework/container/docker_compose.yaml build

rtk env PYTHONPATH="$PWD/tools:$PYTHONPATH" \
ORACLE_LLM_BASE_URL="https://entrypoint/v1" \
ORACLE_LLM_MODEL="Qwen/Qwen3.5-122B-A10B" \
ORACLE_LLM_API_KEY="$ORACLE_LLM_API_KEY" \
ORACLE_TIMEOUT_SEC=60 \
ORACLE_LLM_MAX_TOKENS=1024 \
conda run -n ai_in_robotics python -m autograder.batch.runner \
  --homework 03 \
  --submissions-root /private/tmp/hw3-smoke-submissions \
  --output-root dev/grading_batches \
  --run-id hw3-smoke-1 \
  --max-workers 1
```

Inspect every smoke run before scaling:

- `dev/grading_batches/<run_id>/state.json` should end at `done`.
- `dev/grading_batches/<run_id>/summary.csv` should have one row per student/problem.
- Failed, errored, timed-out, missing, or skipped problems should have diagnostics artifacts.
- Failed or incomplete problems should have feedback markdown/json drafts when the LLM env is configured.
- `tools/data/grades.db` should be unchanged by reports-only batch runs.

DataSchool intake:

```bash
rtk conda run -n ai_in_robotics python dev/scripts/download_dataschool_submissions.py \
  --queue-url "PASTE_FILTERED_DATASCHOOL_QUEUE_URL" \
  --out /private/tmp/ds-hw3-smoke \
  --limit 3 \
  --dry-run \
  --debug-auth

rtk conda run -n ai_in_robotics python dev/scripts/download_dataschool_submissions.py \
  --queue-url "PASTE_FILTERED_DATASCHOOL_QUEUE_URL" \
  --out /private/tmp/ds-hw3-smoke \
  --limit 3

rtk env PYTHONPATH="$PWD/tools:$PYTHONPATH" conda run -n ai_in_robotics python -c "from pathlib import Path; from autograder.batch.job_runner import prepare_dataschool_submissions; print(prepare_dataschool_submissions(Path('/private/tmp/ds-hw3-smoke'), Path('/private/tmp/ds-hw3-prep')))"
```

Recommended scale-up: HW3 with 3 students at `--max-workers 1`, then HW1 with 3 students, then HW2 with 3 students, then 10-20 students at `--max-workers 2`. Use `--max-workers 4` only after Docker memory, runtime, LLM latency, and artifact sizes are stable.

## Student flow

1. Start chat with bot, send `/start`.
2. Ask course questions (text messages) — when oracle is enabled, the bot replies via LLM.
3. Send solution files (.py or .zip) with caption `/grade 01` (or `week 01`).
4. Bot queues the job; autograder runs it in Docker and replies with results (and oracle feedback when enabled).
5. Use `/grades` for your scores, `/leaderboard` for all students.

## Testing

Run tools tests (unit + integration):
```bash
./tools/run_tests.sh
```
Or from `tools/`: `PYTHONPATH="$(pwd):$PYTHONPATH" python -m pytest tests/ -v`

## Structure

```
tools/
├── bot/           # Telegram bot (submissions, /grades, /leaderboard, etc.)
├── autograder/    # Queue, Docker runs, grades DB, pytest parsing, dashboards
├── oracle/        # LLM orchestration (chat, feedback, search)
├── shared/        # Schemas
├── allowlist_db/  # Access control
└── config/        # weeks.yaml, .env.example
```

## Allowlist

- **Env**: Set `ALLOWED_TELEGRAM_IDS=123,456` to restrict by Telegram user ID.
- **SQLite**: If allowlist DB has rows, only those IDs can submit. If empty, all users can (for initial setup). Manage via bot dashboard (port 5001).

## Week registry

- **`tools/config/weeks.yaml`**: minimal; maps week_id → topic_slug.
- **Per-homework config** `*/homework/autograder.yaml`: solution_files, problem_ids, points, metrics, limits (timeout, memory, cpus, network). See `01-intro-and-kinematics/homework/autograder.yaml`.

Tests print `METRIC:problem_id:float_value` for metric leaderboards (e.g. beads bounding sphere).

**Grading run limits**: Time, memory, and CPU are applied via a docker-compose override from `limits` in autograder.yaml. The effective time limit is enforced by the autograder process (subprocess timeout = `timeout_sec + 10`); the container is not sent a separate stop signal. For a hard container-side timeout you would need a wrapper inside the homework image (e.g. `timeout` or a small script that runs pytest under a timer).
