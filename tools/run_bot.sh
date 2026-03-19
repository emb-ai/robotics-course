#!/usr/bin/env bash
# Run the Telegram bot (from repo root, with tools in PYTHONPATH)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools:$PYTHONPATH"
[ -f .env ] && set -a && source .env && set +a
exec python -m bot.main
