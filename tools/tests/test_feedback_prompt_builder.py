"""Tests for TA-only feedback prompt assembly."""

import json
from types import SimpleNamespace

from autograder.batch.artifacts import BatchArtifactStore
from autograder.feedback.prompt_builder import FeedbackPromptContext, build_feedback_prompt


def _write_homework_notebook(repo_root, topic_slug, markdown):
    homework_dir = repo_root / topic_slug / "homework"
    homework_dir.mkdir(parents=True)
    (homework_dir / "homework.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": markdown,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _spec(repo_root, topic_slug="01-intro-and-kinematics"):
    return SimpleNamespace(
        id="01",
        topic_slug=topic_slug,
        homework_dir=str(repo_root / topic_slug / "homework"),
        solution_files=["beads.py", "broom_racing.py"],
        problem_ids={"test_beads.py": "beads"},
        points={"beads": 10.0},
        test_dependencies={"test_beads.py": ["beads.py"]},
    )


def test_prompt_builder_includes_problem_sources_and_diagnostics(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    topic_slug = "01-intro-and-kinematics"
    _write_homework_notebook(repo, topic_slug, "# Beads problem\nCompute the bounding sphere radius.")
    hints_dir = repo / "dev" / topic_slug / "homework" / "feedback_hints"
    hints_dir.mkdir(parents=True)
    (hints_dir / "beads.md").write_text("Check the radius update invariant.\n", encoding="utf-8")
    ref_dir = repo / "dev" / topic_slug / "homework" / "reference_solution"
    ref_dir.mkdir(parents=True)
    (ref_dir / "beads.py").write_text("def solve():\n    return 'reference radius'\n", encoding="utf-8")

    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", "beads.py", "def solve():\n    return 'student radius'\n")
    artifact = store.write_text_artifact(
        "Alice",
        "beads",
        "diagnostic",
        "failure_excerpt.md",
        "AssertionError: radius mismatch\n",
        label="failure_excerpt.md",
    )
    student_result = {
        "student_id": "Alice",
        "submitted_files": ["beads.py"],
        "diagnostics": [
            {
                "plugin_id": "pytest_failure_excerpt",
                "problem_id": "beads",
                "status": "ok",
                "summary": "AssertionError: radius mismatch",
                "artifacts": [artifact],
            }
        ],
    }
    problem_result = {
        "test_file": "test_beads.py",
        "status": "failed",
        "points": 0,
        "max_points": 10,
        "message": "public pytest failure",
    }
    monkeypatch.setenv("API_KEY", "do-not-leak")

    prompt = build_feedback_prompt(
        FeedbackPromptContext(
            repo_root=repo,
            spec=_spec(repo, topic_slug),
            student_result=student_result,
            problem_id="beads",
            problem_result=problem_result,
            artifact_store=store,
        )
    )

    assert "You are drafting feedback for a TA" in prompt.system_prompt
    assert "Compute the bounding sphere radius" in prompt.user_prompt
    assert "Check the radius update invariant" in prompt.user_prompt
    assert "reference radius" in prompt.user_prompt
    assert "student radius" in prompt.user_prompt
    assert "AssertionError: radius mismatch" in prompt.user_prompt
    assert "failure_excerpt.md" in prompt.user_prompt
    assert "public pytest failure" in prompt.user_prompt
    assert "do-not-leak" not in prompt.user_prompt
    assert prompt.metadata["student_files"] == ["beads.py"]
    assert prompt.metadata["reference_files"] == ["beads.py"]
    assert prompt.metadata["diagnostic_artifacts"] == ["failure_excerpt.md"]


def test_prompt_builder_handles_missing_hints_and_reference(tmp_path):
    repo = tmp_path / "repo"
    topic_slug = "01-intro-and-kinematics"
    _write_homework_notebook(repo, topic_slug, "# Beads problem\nShort statement.")
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", "beads.py", "answer = 1\n")

    prompt = build_feedback_prompt(
        FeedbackPromptContext(
            repo_root=repo,
            spec=_spec(repo, topic_slug),
            student_result={"student_id": "Alice", "submitted_files": ["beads.py"], "diagnostics": []},
            problem_id="beads",
            problem_result={"test_file": "test_beads.py", "status": "failed"},
            artifact_store=store,
        )
    )

    assert "## Optional Hints" in prompt.user_prompt
    assert "(none)" in prompt.user_prompt
    assert "## Reference Snippets" in prompt.user_prompt
    assert prompt.metadata["reference_files"] == []


def test_prompt_truncation_is_deterministic_and_keeps_section_headers(tmp_path):
    repo = tmp_path / "repo"
    topic_slug = "01-intro-and-kinematics"
    _write_homework_notebook(repo, topic_slug, "# Beads problem\n" + "large text " * 100)
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", "beads.py", "x = 1\n" + "long_line = 2\n" * 100)
    context = FeedbackPromptContext(
        repo_root=repo,
        spec=_spec(repo, topic_slug),
        student_result={"student_id": "Alice", "submitted_files": ["beads.py"], "diagnostics": []},
        problem_id="beads",
        problem_result={"test_file": "test_beads.py", "status": "failed"},
        artifact_store=store,
        section_char_budget=120,
    )

    first = build_feedback_prompt(context)
    second = build_feedback_prompt(context)

    assert first.user_prompt == second.user_prompt
    assert "## Problem Text" in first.user_prompt
    assert "## Student Code" in first.user_prompt
    assert "...[truncated]" in first.user_prompt
