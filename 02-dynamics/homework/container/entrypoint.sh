#!/usr/bin/env bash
set -e
export PYTEST_CACHE_DIR="${PYTEST_CACHE_DIR:-/tmp/pytest-cache}"
mkdir -p "$PYTEST_CACHE_DIR"

rm -rf /tmp/hw
cp -rL /app/02-dynamics/homework /tmp/hw
cd /tmp/hw

if [ -d "reference_solution" ] && [ -n "$(ls -A reference_solution 2>/dev/null)" ]; then
  if [ "${GRADING_STUDENT_SUBMISSION:-}" != "1" ]; then
    for f in reference_solution/*.py; do
      [ -f "$f" ] || continue
      bn=$(basename "$f")
      [ "$bn" = "__init__.py" ] && continue
      cp "$f" "solutions/$bn"
    done
  fi
  BLOCK_REFERENCE_IMPORT=1 pytest tests/ -v "$@" || true
  r1=$?
  pytest tests/ hidden_tests/ -v --import-mode=importlib "$@"
  r2=$?
  exit $((r1 | r2))
else
  exec pytest tests/ -v "$@"
fi
