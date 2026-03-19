"""Tests for autograder/pytest_parser.py."""

import pytest

from autograder.pytest_parser import parse_metrics, parse_pytest_output

PROBLEM_IDS = {
    "test_beads.py": "beads",
    "test_broom_racing.py": "broom_racing",
    "test_so101_ik.py": "so101_ik",
}


def test_all_passed():
    stdout = """
test_beads.py::test_something PASSED
test_beads.py::test_other PASSED
test_broom_racing.py::test_x PASSED
"""
    result = parse_pytest_output(stdout, "", PROBLEM_IDS)
    assert result["beads"] == 1
    assert result["broom_racing"] == 1
    assert result["so101_ik"] == 0  # no tests run


def test_some_failed():
    stdout = """
test_beads.py::test_a PASSED
test_beads.py::test_b FAILED
test_broom_racing.py::test_x PASSED
"""
    result = parse_pytest_output(stdout, "", PROBLEM_IDS)
    assert result["beads"] == 0
    assert result["broom_racing"] == 1
    assert result["so101_ik"] == 0


def test_error_treated_as_fail():
    stdout = """
test_beads.py::test_a ERROR
test_broom_racing.py::test_x PASSED
"""
    result = parse_pytest_output(stdout, "", PROBLEM_IDS)
    assert result["beads"] == 0
    assert result["broom_racing"] == 1


def test_skipped_ignored():
    stdout = """
test_beads.py::test_a PASSED
test_beads.py::test_b SKIPPED
test_broom_racing.py::test_x PASSED
"""
    result = parse_pytest_output(stdout, "", PROBLEM_IDS)
    assert result["beads"] == 1  # only passed counts
    assert result["broom_racing"] == 1


def test_tests_prefix_in_path():
    stdout = """
tests/test_beads.py::test_x PASSED
"""
    result = parse_pytest_output(stdout, "", PROBLEM_IDS)
    assert result["beads"] == 1


def test_empty_output():
    result = parse_pytest_output("", "", PROBLEM_IDS)
    assert result["beads"] == 0
    assert result["broom_racing"] == 0
    assert result["so101_ik"] == 0


def test_stderr_included():
    stderr = "test_beads.py::test_x PASSED"
    result = parse_pytest_output("", stderr, PROBLEM_IDS)
    assert result["beads"] == 1


def test_import_error_no_tests():
    stdout = """
ImportError while loading ...
"""
    result = parse_pytest_output(stdout, "", PROBLEM_IDS)
    assert result["beads"] == 0
    assert result["broom_racing"] == 0
    assert result["so101_ik"] == 0


def test_parse_metrics():
    stdout = "METRIC:beads:0.42\nMETRIC:broom_racing:1.5\n"
    result = parse_metrics(stdout, "")
    assert result["beads"] == 0.42
    assert result["broom_racing"] == 1.5


def test_parse_metrics_from_stderr():
    result = parse_metrics("", "METRIC:beads:0.123")
    assert result["beads"] == 0.123


def test_parse_metrics_empty():
    assert parse_metrics("", "") == {}
