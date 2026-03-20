#!/usr/bin/env bash
set -e
export PYTEST_CACHE_DIR="${PYTEST_CACHE_DIR:-/tmp/pytest-cache}"
mkdir -p "$PYTEST_CACHE_DIR"

HW_SRC=/app/01-intro-and-kinematics/homework

if [ -d "$HW_SRC/reference_solution" ] && [ -n "$(ls -A "$HW_SRC/reference_solution" 2>/dev/null)" ]; then
  rm -rf /tmp/hw
  cp -rL "$HW_SRC" /tmp/hw
  cd /tmp/hw
  # Autograder bind-mounts student code onto solutions/ (GRADING_STUDENT_SUBMISSION=1).
  # Otherwise (run.sh, local QA): copy reference into solutions/ so tests run the reference.
  if [ "${GRADING_STUDENT_SUBMISSION:-}" != "1" ]; then
    for f in reference_solution/*.py; do
      [ -f "$f" ] || continue
      bn=$(basename "$f")
      [ "$bn" = "__init__.py" ] && continue
      cp "$f" "solutions/$bn"
    done
  fi
  exec pytest tests/ hidden_tests/ -v --import-mode=importlib "$@"
else
  cd "$HW_SRC"
  # When args are given (autograder partial submission), run only those test paths to avoid importing missing solution modules
  if [ $# -gt 0 ]; then
    exec pytest -v "$@"
  else
    exec pytest tests/ -v "$@"
  fi
fi
