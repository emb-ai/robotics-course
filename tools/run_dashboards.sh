#!/usr/bin/env bash
# Run admin dashboards: bot=5001, autograder=5002 (includes grades table and export)
# Access via http://127.0.0.1:5001 etc., or SSH port forward: ssh -L 5001:127.0.0.1:5001 ...
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools:$PYTHONPATH"
[ -f .env ] && set -a && source .env && set +a

echo "Starting dashboards: bot=5001, autograder=5002 (queue, logs, grades)"
python -m bot.dashboard &
P1=$!
python -m autograder.dashboard &
P2=$!

trap "kill $P1 $P2 2>/dev/null; exit" INT TERM
wait
