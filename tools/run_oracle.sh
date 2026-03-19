#!/usr/bin/env bash
# Run the Oracle service (from repo root, with tools in PYTHONPATH)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools:$PYTHONPATH"
[ -f .env ] && set -a && source .env && set +a
exec uvicorn oracle.main:app --host 0.0.0.0 --port 9000
