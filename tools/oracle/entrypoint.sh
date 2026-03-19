#!/bin/bash
# Run SGLang in background, wait for readiness, then start Oracle.
set -e

SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen2-0.5B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
ORACLE_PORT="${ORACLE_PORT:-9000}"
# SGLANG_TP: tensor parallelism degree (number of GPUs). Defaults to 1.
SGLANG_TP="${SGLANG_TP:-1}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

echo "Starting SGLang with model: $SGLANG_MODEL (tp=$SGLANG_TP)"
python3 -m sglang.launch_server \
  --model-path "$SGLANG_MODEL" \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --tp "$SGLANG_TP" \
  $SGLANG_EXTRA_ARGS &
SGLANG_PID=$!

echo "Waiting for SGLang to be ready..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$SGLANG_PORT/v1/models" > /dev/null 2>&1; then
    echo "SGLang is ready."
    break
  fi
  if [ $i -eq 120 ]; then
    echo "SGLang failed to start in time."
    kill $SGLANG_PID 2>/dev/null || true
    exit 1
  fi
  sleep 2
done

# Oracle expects ORACLE_LLM_BASE_URL
export ORACLE_LLM_BASE_URL="${ORACLE_LLM_BASE_URL:-http://localhost:$SGLANG_PORT/v1}"
export ORACLE_LLM_MODEL="${ORACLE_LLM_MODEL:-${SGLANG_MODEL##*/}}"

echo "Starting Oracle on port $ORACLE_PORT"
exec uvicorn oracle.main:app --host 0.0.0.0 --port "$ORACLE_PORT"
