"""Tests for oracle/tools/code_runner.py."""

import pytest

from oracle.tools.code_runner import run_python


def test_simple_code():
    exit_code, stdout, stderr = run_python("print(42)")
    assert exit_code == 0
    assert "42" in stdout
    assert stderr == ""


def test_stderr():
    exit_code, stdout, stderr = run_python("import sys; print('err', file=sys.stderr)")
    assert exit_code == 0
    assert "err" in stderr


def test_exit_nonzero():
    exit_code, stdout, stderr = run_python("exit(3)")
    assert exit_code == 3


def test_timeout():
    exit_code, stdout, stderr = run_python("import time; time.sleep(5)", timeout_sec=1)
    assert exit_code == -1
    assert "timeout" in stderr.lower() or "timed out" in stderr.lower()


def test_stdin():
    exit_code, stdout, stderr = run_python(
        "import sys; print(sys.stdin.read().strip())",
        stdin="hello"
    )
    assert exit_code == 0
    assert "hello" in stdout
