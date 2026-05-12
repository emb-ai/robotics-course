"""Tests for Homework 3 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from autograder.batch.artifacts import BatchArtifactStore
from autograder.diagnostics.base import DiagnosticContext
from autograder.diagnostics.hw03 import (
    HW03RocketEpisodePlugin,
    HW03UnicycleTracePlugin,
    HW03VacmanPathPlugin,
)


TOPIC = "03-control"


def _context(tmp_path, repo_root: Path, problem_id: str, filename: str, source: str) -> DiagnosticContext:
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", filename, source)
    return DiagnosticContext(
        repo_root=repo_root,
        run_dir=store.run_dir,
        topic_slug=TOPIC,
        homework_id="03",
        student_id="Alice",
        problem_id=problem_id,
        test_file=f"test_{problem_id}.py",
        submitted_files=[filename],
        normalized_submission_dir=store.student_dir("Alice") / "submitted",
        student_result={"student_id": "Alice", "submitted_files": [filename]},
        problem_result={"status": "failed", "message": "failed"},
        artifact_store=store,
    )


def test_hw03_rocket_episode_records_notimplemented_trace(tmp_path, repo_root):
    context = _context(
        tmp_path,
        repo_root,
        "rocket",
        "rocket.py",
        "class RocketController:\n"
        "    def reset(self): pass\n"
        "    def __call__(self, obs): raise NotImplementedError('rocket missing')\n",
    )

    result = HW03RocketEpisodePlugin(max_steps=8).run(context)

    assert result.status == "ok"
    assert "notimplemented" in result.summary.lower()
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("rocket_episode.csv") for path in paths)
    assert any(path.endswith("rocket_episode.png") for path in paths)
    assert any(path.endswith("rocket_summary.md") for path in paths)


def test_hw03_vacman_path_writes_episode_json_and_plot(tmp_path, repo_root):
    context = _context(
        tmp_path,
        repo_root,
        "vacman",
        "vacman.py",
        "import numpy as np\n"
        "class VacmanController:\n"
        "    def reset(self): pass\n"
        "    def __call__(self, obs): return np.array([0.0, 0.0])\n",
    )

    result = HW03VacmanPathPlugin(max_steps=8).run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("vacman_episode.json") for path in paths)
    assert any(path.endswith("vacman_path.png") for path in paths)
    assert any(path.endswith("vacman_summary.md") for path in paths)
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("vacman_episode.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert "cleaned" in report


def test_hw03_unicycle_trace_handles_notimplemented(tmp_path, repo_root):
    context = _context(
        tmp_path,
        repo_root,
        "unicycle",
        "unicycle.py",
        "class UnicycleController:\n"
        "    def reset(self): pass\n"
        "    def __call__(self, obs): raise NotImplementedError('unicycle missing')\n",
    )

    result = HW03UnicycleTracePlugin(max_steps=4).run(context)

    assert result.status in {"ok", "skipped"}
    paths = [ref["path"] for ref in result.artifacts]
    if result.status == "ok":
        assert any(path.endswith("unicycle_trace.csv") for path in paths)
        assert any(path.endswith("unicycle_trace.png") for path in paths)
        assert any(path.endswith("unicycle_summary.md") for path in paths)
