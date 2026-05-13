#!/usr/bin/env bash
set -e
export PYTEST_CACHE_DIR="${PYTEST_CACHE_DIR:-/tmp/pytest-cache}"
mkdir -p "$PYTEST_CACHE_DIR"

# When set by the autograder (from homework autograder.yaml limits.timeout_sec), cap pytest
# wall-clock time. Host subprocess timeout is separate (build/pull + this run).
run_pytest() {
  if [ -n "${GRADING_TEST_TIMEOUT_SEC:-}" ] && [ "${GRADING_TEST_TIMEOUT_SEC}" != "0" ]; then
    exec timeout --preserve-status --kill-after=10 "${GRADING_TEST_TIMEOUT_SEC}" "$@"
  else
    exec "$@"
  fi
}

run_selected_student_tests() {
  selected_args=()
  for arg in "$@"; do
    selected_args+=("$arg")
    case "$arg" in
      tests/*.py)
        hidden_test="hidden_tests/${arg#tests/}"
        if [ -f "$hidden_test" ]; then
          selected_args+=("$hidden_test")
        fi
        ;;
    esac
  done
  run_pytest pytest -v --import-mode=importlib "${selected_args[@]}"
}

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
  if [ "${GRADING_STUDENT_SUBMISSION:-}" = "1" ] && [ $# -gt 0 ]; then
    run_selected_student_tests "$@"
  else
    run_pytest pytest tests/ hidden_tests/ -v --import-mode=importlib
  fi
else
  cd "$HW_SRC"
  # When args are given (autograder partial submission), run only those test paths to avoid importing missing solution modules
  if [ $# -gt 0 ]; then
    run_selected_student_tests "$@"
  else
    run_pytest pytest tests/ -v
  fi
fi
