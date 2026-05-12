"""Tests for lightweight generic diagnostics."""

import json

from autograder.batch.artifacts import BatchArtifactStore
from autograder.diagnostics.base import DiagnosticContext
from autograder.diagnostics.generic import (
    MissingDependenciesPlugin,
    PytestFailureExcerptPlugin,
    StaticScanPlugin,
    TimeoutClassifierPlugin,
)


def _context(tmp_path, problem_result=None, submitted_files=None):
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    student_dir = store.student_dir("Alice")
    submitted_dir = student_dir / "submitted"
    for name, content in (submitted_files or {"beads.py": "x = 1\n"}).items():
        store.write_submitted_file("Alice", name, content)
    return DiagnosticContext(
        repo_root=tmp_path,
        run_dir=store.run_dir,
        topic_slug="01-intro-and-kinematics",
        homework_id="01",
        student_id="Alice",
        problem_id="beads",
        test_file="test_beads.py",
        submitted_files=list(submitted_files or {"beads.py": ""}),
        normalized_submission_dir=submitted_dir,
        student_result={
            "student_id": "Alice",
            "submitted_files": list(submitted_files or {"beads.py": ""}),
            "missing_files": ["broom_racing.py"],
        },
        problem_result=problem_result or {"status": "failed", "message": "failed"},
        artifact_store=store,
    )


def test_pytest_failure_excerpt_writes_first_useful_failure_block(tmp_path):
    context = _context(tmp_path)
    student_dir = context.artifact_store.student_dir("Alice")
    (student_dir / "stdout.log").write_text(
        "noise\nFAILED tests/test_beads.py::test_radius - AssertionError\nassert 10 < 2\n",
        encoding="utf-8",
    )

    result = PytestFailureExcerptPlugin().run(context)

    assert result.status == "ok"
    assert "AssertionError" in result.summary
    assert result.artifacts[0]["path"].endswith("failure_excerpt.md")
    excerpt = (context.run_dir / result.artifacts[0]["path"]).read_text()
    assert "test_radius" in excerpt


def test_static_scan_reports_syntax_errors_and_forbidden_imports(tmp_path):
    context = _context(
        tmp_path,
        submitted_files={
            "bad.py": "def broken(:\n",
            "leak.py": "from reference_solution import answer\nimport hidden_tests.secret\n",
        },
    )

    result = StaticScanPlugin().run(context)

    assert result.status == "ok"
    assert result.metrics["syntax_errors"] == 1
    assert result.metrics["forbidden_imports"] == 2
    json_ref = next(ref for ref in result.artifacts if ref["path"].endswith("static_scan.json"))
    report = json.loads((context.run_dir / json_ref["path"]).read_text())
    assert report["syntax_errors"][0]["file"] == "bad.py"
    assert {item["module"] for item in report["forbidden_imports"]} == {
        "reference_solution",
        "hidden_tests.secret",
    }


def test_missing_dependencies_report_includes_expected_and_submitted_files(tmp_path):
    context = _context(
        tmp_path,
        problem_result={
            "status": "missing",
            "message": "Missing dependencies: beads.py, broom_racing.py",
        },
        submitted_files={"beads.py": "x = 1\n"},
    )

    result = MissingDependenciesPlugin().run(context)

    assert result.status == "ok"
    report = (context.run_dir / result.artifacts[0]["path"]).read_text()
    assert "broom_racing.py" in report
    assert "beads.py" in report


def test_timeout_classifier_writes_timeout_summary(tmp_path):
    context = _context(
        tmp_path,
        problem_result={"status": "timeout", "message": "subprocess timed out"},
    )
    student_dir = context.artifact_store.student_dir("Alice")
    (student_dir / "stderr.log").write_text("TimeoutExpired after 120 seconds\n", encoding="utf-8")

    result = TimeoutClassifierPlugin().run(context)

    assert result.status == "ok"
    assert "timeout" in result.summary.lower()
    assert result.artifacts[0]["path"].endswith("timeout.md")
