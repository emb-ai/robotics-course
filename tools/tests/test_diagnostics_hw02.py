"""Tests for Homework 2 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from autograder.batch.artifacts import BatchArtifactStore
from autograder.diagnostics.base import DiagnosticContext
from autograder.diagnostics.hw02 import (
    HW02ConstraintDriftPlugin,
    HW02KinEnergyTracePlugin,
    HW02PenaltyForcePlugin,
)


TOPIC = "02-dynamics"


def _context(tmp_path, repo_root: Path, problem_id: str, files: dict[str, str]) -> DiagnosticContext:
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    for name, source in files.items():
        store.write_submitted_file("Alice", name, source)
    return DiagnosticContext(
        repo_root=repo_root,
        run_dir=store.run_dir,
        topic_slug=TOPIC,
        homework_id="02",
        student_id="Alice",
        problem_id=problem_id,
        test_file=f"test_{problem_id}.py",
        submitted_files=sorted(files),
        normalized_submission_dir=store.student_dir("Alice") / "submitted",
        student_result={"student_id": "Alice", "submitted_files": sorted(files)},
        problem_result={"status": "failed", "message": "failed"},
        artifact_store=store,
    )


def test_hw02_kin_energy_trace_writes_csv_png_and_markdown(tmp_path):
    context = _context(
        tmp_path,
        tmp_path / "repo",
        "kin_energy",
        {
            "kin_energy.py": (
                "class KinEnergyCallback:\n"
                "    def __init__(self, draw_graph=False): pass\n"
                "    def calc_kinetic_energy(self, bodies):\n"
                "        return sum(getattr(body, 'kinetic_energy', 0.0) for body in bodies)\n"
            ),
            "ode_solvers.py": "class EulerMethod: pass\nclass RK4Method: pass\n",
        },
    )

    result = HW02KinEnergyTracePlugin().run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("kin_energy_trace.csv") for path in paths)
    assert any(path.endswith("kin_energy_trace.png") for path in paths)
    assert any(path.endswith("kin_energy_summary.md") for path in paths)
    assert result.metrics["nonfinite_values"] == 0


def test_hw02_constraint_drift_records_api_failures(tmp_path):
    context = _context(
        tmp_path,
        tmp_path / "repo",
        "joints",
        {
            "constraints.py": (
                "class BallAndSocketJoint:\n"
                "    def __init__(self, *args, **kwargs): pass\n"
                "    def get_C(self): raise NotImplementedError('missing C')\n"
                "    def get_J_updates(self): raise NotImplementedError('missing J')\n"
            ),
            "constraints_manager.py": (
                "class ConstraintsManagerImpulseBased:\n"
                "    def __init__(self, *args, **kwargs): pass\n"
                "    def calc_forces(self, *args): raise NotImplementedError('missing forces')\n"
            ),
            "ode_solvers.py": "class RK4Method: pass\n",
        },
    )

    result = HW02ConstraintDriftPlugin().run(context)

    assert result.status == "ok"
    assert "notimplemented" in result.summary.lower() or "missing" in result.summary.lower()
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("constraint_drift.csv") for path in paths)
    assert any(path.endswith("constraint_drift.png") for path in paths)
    assert any(path.endswith("constraint_drift_summary.md") for path in paths)


def test_hw02_penalty_force_writes_case_report(tmp_path):
    context = _context(
        tmp_path,
        tmp_path / "repo",
        "penalty",
        {
            "penalty.py": (
                "import numpy as np\n"
                "class PenaltyMethod:\n"
                "    def __init__(self, *args, **kwargs): pass\n"
                "    def step(self, collisions, t0, t1):\n"
                "        for col in collisions:\n"
                "            col.obj_a.force_accumulator += np.array([-1.0, -2.0, 0.0])\n"
                "            col.obj_a.torque_accumulator += np.array([0.0, 0.0, -1.0])\n"
            ),
        },
    )

    result = HW02PenaltyForcePlugin().run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("penalty_cases.json") for path in paths)
    assert any(path.endswith("penalty_force_summary.md") for path in paths)
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("penalty_cases.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert report["cases"]
