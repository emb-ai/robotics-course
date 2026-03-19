#!/bin/bash
# Launched by cloud.ru Distributed Train (binary job). mpirun executes this path inside the image.
# - Oracle app: /opt/oracle/tools (baked at build)
# - Course repo: /opt/course-repo (clone in image + optional git pull at start)
set -e

SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
ORACLE_PORT="${ORACLE_PORT:-9000}"
SGLANG_TP="${SGLANG_TP:-8}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

ORACLE_DIR="${ORACLE_DIR:-/opt/oracle}"
REPO_DIR="${REPO_DIR:-/opt/course-repo}"
COURSE_REPO_URL="${COURSE_REPO_URL:-https://github.com/emb-ai/robotics-course.git}"
COURSE_REPO_BRANCH="${COURSE_REPO_BRANCH:-2026}"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Course repo: updating $REPO_DIR (git pull origin $COURSE_REPO_BRANCH)..."
  if ! git -C "$REPO_DIR" pull --ff-only origin "$COURSE_REPO_BRANCH" 2>/dev/null; then
    echo "Note: course repo pull skipped (offline, shallow, or diverged); using tree on disk."
  fi
else
  echo "Course repo: cloning $COURSE_REPO_URL ($COURSE_REPO_BRANCH) -> $REPO_DIR ..."
  rm -rf "$REPO_DIR"
  git clone --depth 1 --branch "$COURSE_REPO_BRANCH" "$COURSE_REPO_URL" "$REPO_DIR"
fi

export PYTHONPATH="$ORACLE_DIR/tools:$PYTHONPATH"
export AI_ROBOTICS_REPO_ROOT="$REPO_DIR"
export ORACLE_LLM_BASE_URL="http://localhost:$SGLANG_PORT/v1"
export ORACLE_LLM_MODEL="${ORACLE_LLM_MODEL:-${SGLANG_MODEL##*/}}"

echo "Starting SGLang: model=$SGLANG_MODEL tp=$SGLANG_TP port=$SGLANG_PORT"
python3 -m sglang.launch_server \
    --model-path "$SGLANG_MODEL" \
    --host 0.0.0.0 --port "$SGLANG_PORT" \
    --tp "$SGLANG_TP" \
    $SGLANG_EXTRA_ARGS &
SGLANG_PID=$!

echo "Waiting for SGLang to be ready (up to 600s) ..."
for i in $(seq 1 300); do
    if curl -sf "http://localhost:$SGLANG_PORT/v1/models" > /dev/null 2>&1; then
        echo "SGLang ready."
        break
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: SGLang did not start in time."
        kill "$SGLANG_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

echo "Starting Oracle on port $ORACLE_PORT (repo root=$REPO_DIR)"
cd "$ORACLE_DIR"
uvicorn oracle.main:app --host 0.0.0.0 --port "$ORACLE_PORT" &
ORACLE_PID=$!

echo "Services running. SSH tunnel from V100:"
echo "  ssh -N -L $ORACLE_PORT:localhost:$ORACLE_PORT <this-host> -p 2222 -i <key>"

wait "$SGLANG_PID" "$ORACLE_PID"
