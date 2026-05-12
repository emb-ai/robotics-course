"""Tests for batch feedback draft generation."""

import importlib
import json
from types import SimpleNamespace

from autograder.batch.artifacts import BatchArtifactStore
from autograder.feedback.batch_feedback import generate_feedback_for_student
from autograder.feedback.client import FeedbackResult


def _spec(repo_root):
    topic_slug = "01-intro-and-kinematics"
    homework_dir = repo_root / topic_slug / "homework"
    homework_dir.mkdir(parents=True)
    (homework_dir / "homework.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": "# Beads\nProblem text."}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        id="01",
        topic_slug=topic_slug,
        homework_dir=str(homework_dir),
        solution_files=["beads.py"],
        problem_ids={"test_beads.py": "beads", "test_broom.py": "broom"},
        points={"beads": 10.0, "broom": 5.0},
        test_dependencies={"test_beads.py": ["beads.py"], "test_broom.py": ["broom.py"]},
    )


def test_generate_feedback_for_student_writes_markdown_and_metadata(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    spec = _spec(repo)
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", "beads.py", "answer = 1\n")
    student_result = {
        "student_id": "Alice",
        "submitted_files": ["beads.py"],
        "problems": {
            "beads": {
                "test_file": "test_beads.py",
                "status": "failed",
                "points": 0,
                "max_points": 10,
                "message": "bad radius",
            },
            "broom": {
                "test_file": "test_broom.py",
                "status": "passed",
                "points": 5,
                "max_points": 5,
            },
        },
        "diagnostics": [
            {
                "plugin_id": "demo",
                "problem_id": "beads",
                "status": "ok",
                "summary": "radius mismatch",
                "artifacts": [],
            }
        ],
        "artifacts": [],
    }

    def fake_client(prompt, config=None):
        return FeedbackResult(status="ok", content="Likely radius update issue.", model="demo-model")

    monkeypatch.setattr("autograder.feedback.batch_feedback.generate_feedback", fake_client)

    refs = generate_feedback_for_student(repo, spec, student_result, store)

    assert len(refs) == 2
    assert student_result["feedback"][0]["status"] == "ok"
    assert student_result["artifacts"] == refs
    md_path = store.run_dir / "students" / "Alice" / "feedback" / "beads.md"
    json_path = store.run_dir / "students" / "Alice" / "feedback" / "beads.json"
    assert md_path.is_file()
    assert json_path.is_file()
    markdown = md_path.read_text(encoding="utf-8")
    assert "TA REVIEW ONLY" in markdown
    assert "Likely radius update issue." in markdown
    assert "demo-model" in markdown
    metadata = json.loads(json_path.read_text(encoding="utf-8"))["metadata"]
    assert metadata["status"] == "ok"
    assert metadata["manifest"]["diagnostics"] == ["demo: radius mismatch"]


def test_generate_feedback_for_student_writes_error_metadata_without_blocking(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    spec = _spec(repo)
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", "beads.py", "answer = 1\n")
    student_result = {
        "student_id": "Alice",
        "submitted_files": ["beads.py"],
        "problems": {
            "beads": {"test_file": "test_beads.py", "status": "failed", "message": "bad radius"}
        },
        "diagnostics": [],
        "artifacts": [],
    }

    monkeypatch.setattr(
        "autograder.feedback.batch_feedback.generate_feedback",
        lambda prompt, config=None: FeedbackResult(status="error", error="HTTP 500", model="demo-model"),
    )

    refs = generate_feedback_for_student(repo, spec, student_result, store)

    assert len(refs) == 2
    metadata = json.loads(
        (store.run_dir / "students" / "Alice" / "feedback" / "beads.json").read_text(encoding="utf-8")
    )["metadata"]
    assert metadata["status"] == "error"
    assert metadata["error"] == "HTTP 500"


def test_feedback_package_does_not_import_dashboard_worker_or_grades_store():
    importlib.import_module("autograder.feedback.batch_feedback")

    forbidden = {
        "autograder.dashboard",
        "autograder.worker",
        "autograder.grades.store",
        "bot.dashboard",
    }
    imported = set(importlib.sys.modules)
    assert forbidden.isdisjoint(imported)
