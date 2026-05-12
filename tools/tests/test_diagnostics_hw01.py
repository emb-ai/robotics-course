"""Tests for Homework 1 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from autograder.batch.artifacts import BatchArtifactStore
from autograder.diagnostics.base import DiagnosticContext
from autograder.diagnostics.hw01 import (
    HW01BeadsComparePlugin,
    HW01BroomRacingTrajectoryPlugin,
    HW01SO101IKDebugPlugin,
)


TOPIC = "01-intro-and-kinematics"


def _context(tmp_path, repo_root: Path, problem_id: str, filename: str, source: str) -> DiagnosticContext:
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", filename, source)
    return DiagnosticContext(
        repo_root=repo_root,
        run_dir=store.run_dir,
        topic_slug=TOPIC,
        homework_id="01",
        student_id="Alice",
        problem_id=problem_id,
        test_file=f"test_{problem_id}.py",
        submitted_files=[filename],
        normalized_submission_dir=store.student_dir("Alice") / "submitted",
        student_result={"student_id": "Alice", "submitted_files": [filename]},
        problem_result={"status": "failed", "message": "failed"},
        artifact_store=store,
    )


def _write_hw01_reference(repo_root: Path, filename: str, source: str) -> None:
    ref_dir = repo_root / "dev" / TOPIC / "homework" / "reference_solution"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "__init__.py").write_text("", encoding="utf-8")
    (ref_dir / filename).write_text(source, encoding="utf-8")


def test_hw01_beads_compare_writes_case_summary_and_plot(tmp_path):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.full((len(link_lengths) - 1, 2), 0.2)\n",
    )

    result = HW01BeadsComparePlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("beads_case.json") for path in paths)
    assert any(path.endswith("beads_summary.md") for path in paths)
    assert any(path.endswith("beads_radius.png") for path in paths)
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("beads_case.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert {"student_radius", "reference_radius", "delta", "links"} <= set(report)


def test_hw01_beads_compare_records_slow_student_timeout(tmp_path):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import time, numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    time.sleep(2)\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )

    result = HW01BeadsComparePlugin(case_timeout_sec=0.05).run(context)

    assert result.status == "ok"
    assert "timeout" in result.summary.lower()
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("beads_case.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert report["error"] and "timeout" in report["error"].lower()


def test_hw01_broom_trajectory_writes_worst_cases_and_plots(tmp_path):
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / TOPIC / "homework" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "broom_racing_open_test_cases.json").write_text(
        json.dumps(
            [
                {"task": "gate_pass", "start": [0, 0, 0, 0, 0], "goal": [1, 0, 0, 0, 0]},
                {"task": "catch_snitch", "start": [0, 0, 0, 0, 0], "goal_xyz": [1, 1, 0]},
                {
                    "task": "catch_ball_and_gate",
                    "start": [0, 0, 0, 0, 0],
                    "goal_xyz": [0.5, 0.5, 0],
                    "goal": [1, 1, 0, 0, 0],
                },
            ]
        ),
        encoding="utf-8",
    )
    curve_source = (
        "import numpy as np\n"
        "class Curve:\n"
        "    def __init__(self, sx, sy, sz, gx, gy, gz):\n"
        "        self.sx=sx; self.sy=sy; self.sz=sz; self.gx=gx; self.gy=gy; self.gz=gz\n"
        "    def __call__(self, s):\n"
        "        t=float(np.asarray(s).flat[0])\n"
        "        return type('C', (), {'x': self.sx + (self.gx-self.sx)*t, 'y': self.sy + (self.gy-self.sy)*t, 'z': self.sz + (self.gz-self.sz)*t, 'theta': 0.0, 'phi': 0.0})()\n"
        "def gate_pass_ref(start, goal): return Curve(start.x,start.y,start.z,goal.x,goal.y,goal.z)\n"
        "def catch_snitch_ref(start, goal): return Curve(start.x,start.y,start.z,goal.x,goal.y,goal.z)\n"
        "def catch_ball_and_gate_ref(start, intermediate, final): return Curve(start.x,start.y,start.z,final.x,final.y,final.z)\n"
    )
    _write_hw01_reference(repo_root, "broom_racing.py", curve_source)
    context = _context(
        tmp_path,
        repo_root,
        "broom_racing",
        "broom_racing.py",
        curve_source.replace("_ref", ""),
    )

    result = HW01BroomRacingTrajectoryPlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("broom_worst_cases.json") for path in paths)
    assert any(path.endswith("broom_summary.md") for path in paths)
    assert any(path.endswith(".png") for path in paths)


def test_hw01_so101_debug_classifies_missing_and_nonfinite_cases(tmp_path):
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / TOPIC / "homework" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "so101_ik_open_test_cases.json").write_text(
        json.dumps([{"pose": [0.2, 0.0, 0.04, 0.0], "solvable_analytical": True, "solvable_numerical": True}]),
        encoding="utf-8",
    )
    context = _context(
        tmp_path,
        repo_root,
        "so101_ik",
        "so101_ik.py",
        "import sympy\n"
        "def numerical_ik_so101_downturned(x, y, z, yaw): return None\n"
        "def analytical_ik_so101_downturned(x, y, z, yaw): return {'shoulder_pan': float('nan')}\n"
        "def so101_downturned_ik_symbolic(x, y, z, yaw): return {'shoulder_pan': sympy.nan}\n",
    )

    result = HW01SO101IKDebugPlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    assert "numerical" in result.summary.lower()
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("so101_failures.json") for path in paths)
    assert any(path.endswith("so101_summary.md") for path in paths)
