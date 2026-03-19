#!/usr/bin/env bash
# Run tools tests (from repo root or tools/)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools:$PYTHONPATH"
exec python -m pytest tools/tests/ -v "$@"
