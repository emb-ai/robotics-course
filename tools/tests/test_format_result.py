"""Tests for autograder/worker.py format_result."""

from autograder.worker import format_result


def test_exit_code_zero_short():
    msg, doc = format_result(0, "ok", "", max_inline=3500)
    assert "passed" in msg.lower()
    assert doc is None


def test_exit_code_nonzero_short():
    msg, doc = format_result(1, "x", "y", max_inline=3500)
    assert "fail" in msg.lower()
    assert doc is None


def test_long_output_attaches_doc():
    long_out = "x" * 4000
    msg, doc = format_result(0, long_out, "", max_inline=3500)
    assert "attached" in msg.lower()
    assert doc is not None
    assert len(doc) > 3500


def test_first_failure_snippet():
    stdout = """
test_a PASSED
test_b FAILED
some error message
"""
    msg, _ = format_result(1, stdout, "", max_inline=3500)
    assert "FAILED" in msg or "First failure" in msg or "fail" in msg.lower()
