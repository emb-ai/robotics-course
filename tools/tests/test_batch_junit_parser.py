"""Tests for JUnit XML parsing in batch grading."""

from autograder.batch.junit_parser import parse_junit_results
from autograder.batch.models import ProblemRunResult


PROBLEM_IDS = {
    "test_beads.py": "beads",
    "test_broom.py": "broom",
    "test_error.py": "error_problem",
    "test_skip.py": "skip_problem",
}
POINTS = {"beads": 4, "broom": 6, "error_problem": 2, "skip_problem": 1, "missing": 3}


def test_junit_parser_maps_pass_fail_error_and_skipped_to_problem_results():
    xml = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest">
    <testcase classname="tests.test_beads" name="test_a" file="tests/test_beads.py" />
    <testcase classname="tests.test_beads" name="test_b" file="tests/test_beads.py" />
    <testcase classname="tests.test_broom" name="test_x" file="tests/test_broom.py">
      <failure message="bad radius">assert False</failure>
    </testcase>
    <testcase classname="tests.test_error" name="test_x" file="tests/test_error.py">
      <error message="import failed">ImportError</error>
    </testcase>
    <testcase classname="tests.test_skip" name="test_x" file="tests/test_skip.py">
      <skipped message="not relevant" />
    </testcase>
  </testsuite>
</testsuites>
"""

    results = parse_junit_results(xml, PROBLEM_IDS, POINTS)
    by_problem = {result.problem_id: result for result in results}

    assert by_problem["beads"].status == "passed"
    assert by_problem["beads"].points_earned == 4
    assert by_problem["broom"].status == "failed"
    assert by_problem["broom"].points_earned == 0
    assert by_problem["error_problem"].status == "error"
    assert by_problem["skip_problem"].status == "skipped"


def test_junit_parser_keeps_missing_results_for_unrun_tests():
    missing = ProblemRunResult(
        problem_id="missing",
        test_file="test_missing.py",
        status="missing",
        points_possible=3,
        points_earned=0,
        message="Missing dependencies: answer.py",
    )

    results = parse_junit_results("", {"test_missing.py": "missing"}, POINTS, missing_results=[missing])

    assert results == [missing]
