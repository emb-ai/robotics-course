#!/usr/bin/env bash
set -e
export PYTEST_CACHE_DIR="${PYTEST_CACHE_DIR:-/tmp/pytest-cache}"
mkdir -p "$PYTEST_CACHE_DIR"

# When set by the autograder (limits.timeout_sec), cap each pytest phase wall time.
timeout_wrap() {
  if [ -n "${GRADING_TEST_TIMEOUT_SEC:-}" ] && [ "${GRADING_TEST_TIMEOUT_SEC}" != "0" ]; then
    timeout --preserve-status --kill-after=10 "${GRADING_TEST_TIMEOUT_SEC}" "$@"
  else
    "$@"
  fi
}

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
  if [ "${GRADING_STUDENT_SUBMISSION:-}" = "1" ] && [ $# -gt 0 ]; then
    run_selected_student_tests "$@"
  else
    BLOCK_REFERENCE_IMPORT=1 timeout_wrap pytest tests/ -v || true
    r1=$?
    timeout_wrap pytest tests/ hidden_tests/ -v --import-mode=importlib
    r2=$?
    exit $((r1 | r2))
  fi
else
  if [ "${GRADING_STUDENT_SUBMISSION:-}" = "1" ] && [ $# -gt 0 ]; then
    run_selected_student_tests "$@"
  else
    run_pytest pytest tests/ -v
  fi
fi
