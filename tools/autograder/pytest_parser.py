"""Parse pytest -v output to extract per-problem pass/fail (binary) and METRIC: lines."""

import re
from typing import Any

# Pattern: METRIC:problem_id:float_value
METRIC_PATTERN = re.compile(r"METRIC:(\w+):([+-]?\d*\.?\d+)", re.IGNORECASE)


def parse_metrics(stdout: str, stderr: str) -> dict[str, float]:
    """
    Extract METRIC:problem_id:value lines from output.
    Returns dict: problem_id -> float.
    """
    text = stdout + "\n" + stderr
    result: dict[str, float] = {}
    for m in METRIC_PATTERN.finditer(text):
        problem_id, val_str = m.group(1), m.group(2)
        try:
            result[problem_id] = float(val_str)
        except ValueError:
            continue
    return result


def parse_pytest_output(
    stdout: str,
    stderr: str,
    problem_ids: dict[str, str],
) -> dict[str, int]:
    """
    Map test file basename -> problem_id. Each problem passes iff ALL tests in that file pass.
    Returns dict: problem_id -> 0 or 1.
    """
    text = stdout + "\n" + stderr

    # Collect per-file results: test_file -> set of "passed" | "failed" | "skipped"
    # Pattern: test_beads.py::test_X PASSED or tests/test_beads.py::... or hidden_tests/test_beads.py::...
    file_results: dict[str, set[str]] = {}
    pattern = re.compile(
        r"(?:^|\n)\s*(?:(?:tests|hidden_tests)/)?(\w+\.py)::[\w:]+ (PASSED|FAILED|ERROR|SKIPPED)(?:\s|$)",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        fname = m.group(1)
        status = m.group(2).lower()
        if status == "error":
            status = "failed"
        if status == "skipped":
            # SKIPPED counts as neither pass nor fail for grading; don't add to statuses
            continue
        if fname not in file_results:
            file_results[fname] = set()
        file_results[fname].add(status)

    result: dict[str, int] = {}
    for test_file, problem_id in problem_ids.items():
        statuses = file_results.get(test_file, set())
        # Pass iff all tests in that file passed. No tests / any failed / ERROR = fail.
        passed = 1 if statuses == {"passed"} else 0
        result[problem_id] = passed

    return result
