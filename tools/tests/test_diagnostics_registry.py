"""Tests for diagnostics plugin registry behavior."""

import time

from autograder.batch.artifacts import BatchArtifactStore
from autograder.diagnostics.base import DiagnosticContext, DiagnosticResult
from autograder.diagnostics.registry import DiagnosticRegistry, should_run_diagnostics
from autograder.diagnostics.run_diagnostics import run_diagnostics_for_student


class _Plugin:
    id = "demo"
    label = "Demo"
    timeout_sec = 1.0

    def __init__(self, supported=True, result_status="ok"):
        self.supported = supported
        self.result_status = result_status

    def supports(self, context):
        return self.supported

    def run(self, context):
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status=self.result_status,
            summary="ran",
        )


class _FailingPlugin(_Plugin):
    id = "failing"

    def run(self, context):
        raise RuntimeError("boom")


class _SlowPlugin(_Plugin):
    id = "slow"
    timeout_sec = 0.01

    def run(self, context):
        time.sleep(0.05)
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary="too late",
        )


def _context(tmp_path, status="failed"):
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    return DiagnosticContext(
        repo_root=tmp_path,
        run_dir=store.run_dir,
        topic_slug="01-intro-and-kinematics",
        homework_id="01",
        student_id="Alice",
        problem_id="beads",
        test_file="test_beads.py",
        submitted_files=["beads.py"],
        normalized_submission_dir=store.student_dir("Alice") / "submitted",
        student_result={"student_id": "Alice"},
        problem_result={"status": status},
        artifact_store=store,
    )


def test_registry_returns_only_supported_plugins(tmp_path):
    registry = DiagnosticRegistry()
    supported = _Plugin(supported=True)
    unsupported = _Plugin(supported=False)
    unsupported.id = "other"
    registry.register(supported)
    registry.register(unsupported)

    selected = registry.select(_context(tmp_path))

    assert selected == [supported]


def test_duplicate_plugin_ids_are_rejected():
    registry = DiagnosticRegistry()
    registry.register(_Plugin())

    try:
        registry.register(_Plugin())
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate plugin id to fail")


def test_failed_plugin_returns_error_result_without_raising(tmp_path):
    registry = DiagnosticRegistry([_FailingPlugin()])

    results = registry.run(_context(tmp_path))

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].plugin_id == "failing"
    assert "boom" in (results[0].error or "")


def test_slow_plugin_returns_timeout_result(tmp_path):
    registry = DiagnosticRegistry([_SlowPlugin()])

    results = registry.run(_context(tmp_path))

    assert len(results) == 1
    assert results[0].status == "timeout"
    assert results[0].plugin_id == "slow"


def test_should_run_diagnostics_only_for_non_passed_problem_statuses():
    assert should_run_diagnostics({"status": "failed"})
    assert should_run_diagnostics({"status": "error"})
    assert should_run_diagnostics({"status": "timeout"})
    assert should_run_diagnostics({"status": "missing"})
    assert should_run_diagnostics({"status": "skipped"})
    assert not should_run_diagnostics({"status": "passed"})


def test_run_diagnostics_for_student_skips_passed_problems(tmp_path):
    registry = DiagnosticRegistry([_Plugin()])
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", "beads.py", "x = 1\n")
    student_result = {
        "student_id": "Alice",
        "submitted_files": ["beads.py"],
        "problems": {
            "beads": {"status": "passed", "test_file": "test_beads.py"},
            "broom": {"status": "failed", "test_file": "test_broom.py"},
        },
    }

    results = run_diagnostics_for_student(
        repo_root=tmp_path,
        spec={"id": "01", "topic_slug": "01-intro-and-kinematics"},
        student_result=student_result,
        artifact_store=store,
        registry=registry,
    )

    assert len(results) == 1
    assert results[0]["problem_id"] == "broom"
