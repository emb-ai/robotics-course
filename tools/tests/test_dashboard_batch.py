"""Tests for the file-backed batch grading dashboard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def batch_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOGRADER_BATCH_OUTPUT_ROOT", str(tmp_path / "batches"))
    from autograder.dashboard import app

    app.config.update(TESTING=True)
    return app.test_client()


def _write_run(root: Path, run_id: str, *, homework_id: str = "01", status: str = "done") -> Path:
    run_dir = root / run_id
    student_dir = run_dir / "students" / "Alice_Example"
    (student_dir / "submitted").mkdir(parents=True)
    (student_dir / "diagnostics" / "beads").mkdir(parents=True)
    (student_dir / "feedback").mkdir(parents=True)
    (student_dir / "submitted" / "beads.py").write_text("answer = 1\n", encoding="utf-8")
    (student_dir / "stdout.log").write_text("container stdout", encoding="utf-8")
    (student_dir / "stderr.log").write_text("container stderr", encoding="utf-8")
    (student_dir / "pytest.xml").write_text("<testsuite />", encoding="utf-8")
    (student_dir / "diagnostics" / "beads" / "trace.txt").write_text("radius mismatch", encoding="utf-8")
    (student_dir / "feedback" / "beads.md").write_text("TA review draft", encoding="utf-8")
    (student_dir / "student.json").write_text(
        json.dumps({"student_id": "Alice Example", "student_path_id": "Alice_Example"}),
        encoding="utf-8",
    )
    artifacts = [
        {
            "kind": "diagnostic",
            "label": "trace",
            "problem_id": "beads",
            "path": "students/Alice_Example/diagnostics/beads/trace.txt",
        },
        {
            "kind": "feedback",
            "label": "beads.md",
            "problem_id": "beads",
            "path": "students/Alice_Example/feedback/beads.md",
        },
    ]
    (student_dir / "artifacts.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    result = {
        "student_id": "Alice Example",
        "student_path_id": "Alice_Example",
        "status": "failed",
        "submitted_files": ["beads.py"],
        "problems": {
            "beads": {
                "test_file": "test_beads.py",
                "status": "failed",
                "points": 0,
                "max_points": 10,
                "message": "bad radius",
            }
        },
        "diagnostics": [{"problem_id": "beads", "summary": "radius mismatch"}],
        "feedback": [{"problem_id": "beads", "status": "ok"}],
        "metrics": {"beads": 2.0},
        "artifacts": artifacts,
    }
    (student_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    batch_result = {
        "homework_id": homework_id,
        "problem_ids": ["beads"],
        "points": {"beads": 10},
        "students": [result],
    }
    (run_dir / "results.json").write_text(json.dumps(batch_result), encoding="utf-8")
    (run_dir / "summary.csv").write_text(
        "student_id,student_path_id,problem_id,status,points,max_points\n"
        "Alice Example,Alice_Example,beads,failed,0,10\n",
        encoding="utf-8",
    )
    (run_dir / "index.html").write_text("<h1>Static report</h1>", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"homework_id": homework_id}), encoding="utf-8")
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": status,
                "counts": {"total": 1, "queued": 0, "running": 0, "completed": 1, "failed": 1},
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:02:00+00:00",
                "students": {"Alice Example": "done"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_batches_index_renders_without_redis(batch_client):
    response = batch_client.get("/batches")

    assert response.status_code == 200
    assert b"Batch grading" in response.data
    assert b"No batch runs" in response.data


def test_new_batch_form_renders_homeworks_and_source_modes(batch_client):
    response = batch_client.get("/batches/new")

    assert response.status_code == 200
    assert b"Local submissions root" in response.data
    assert b"DataSchool" in response.data
    assert b"Max workers" in response.data


def test_start_batch_rejects_invalid_inputs(batch_client, tmp_path):
    missing = batch_client.post(
        "/batches/new",
        data={"run_id": "../bad", "homework_id": "01", "source_mode": "local", "max_workers": "2"},
    )
    assert missing.status_code == 400
    assert b"unsafe run id" in missing.data.lower()

    bad_workers = batch_client.post(
        "/batches/new",
        data={
            "run_id": "run-01",
            "homework_id": "01",
            "source_mode": "local",
            "submissions_root": str(tmp_path / "missing"),
            "max_workers": "0",
        },
    )
    assert bad_workers.status_code == 400
    assert b"max workers" in bad_workers.data.lower()


def test_start_batch_local_launches_job_runner(batch_client, tmp_path, monkeypatch):
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    popen_calls = []

    class FakePopen:
        pid = 4321

        def __init__(self, cmd, **kwargs):
            popen_calls.append((cmd, kwargs))

    monkeypatch.setattr("autograder.dashboard_batch.subprocess.Popen", FakePopen)

    response = batch_client.post(
        "/batches/new",
        data={
            "run_id": "run-01",
            "homework_id": "01",
            "source_mode": "local",
            "submissions_root": str(submissions),
            "max_workers": "4",
            "enable_diagnostics": "on",
        },
    )

    assert response.status_code == 302
    assert popen_calls
    cmd = popen_calls[0][0]
    assert cmd[:3] == [sys.executable, "-m", "autograder.batch.job_runner"]
    assert "--job-config" in cmd
    job_path = Path(cmd[cmd.index("--job-config") + 1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["submissions_root"] == str(submissions)
    assert job["enable_diagnostics"] is True
    assert job["enable_feedback"] is False
    assert "DATASCHOOL_COOKIE" not in json.dumps(job)


def test_start_dataschool_requires_cookie_env_or_file(batch_client, monkeypatch):
    monkeypatch.delenv("DATASCHOOL_COOKIE", raising=False)

    response = batch_client.post(
        "/batches/new",
        data={
            "run_id": "run-01",
            "homework_id": "01",
            "source_mode": "dataschool",
            "queue_url": "https://lk.dataschool.yandex.ru/teaching/assignments/?course=1704",
            "cookie_file": "",
            "max_workers": "2",
        },
    )

    assert response.status_code == 400
    assert b"cookie" in response.data.lower()


def test_run_and_student_detail_render_artifacts_and_feedback(batch_client, tmp_path, monkeypatch):
    output_root = Path(os.environ["AUTOGRADER_BATCH_OUTPUT_ROOT"])
    _write_run(output_root, "run-01")

    run_response = batch_client.get("/batches/run-01")
    assert run_response.status_code == 200
    assert b"Alice Example" in run_response.data
    assert b"radius mismatch" in run_response.data
    assert b"Feedback" in run_response.data

    student_response = batch_client.get("/batches/run-01/students/Alice_Example")
    assert student_response.status_code == 200
    assert b"container stdout" in student_response.data
    assert b"bad radius" in student_response.data
    assert b"trace.txt" in student_response.data
    assert b"TA review draft" in student_response.data


def test_exports_and_raw_artifact_route_are_path_safe(batch_client, tmp_path, monkeypatch):
    output_root = Path(os.environ["AUTOGRADER_BATCH_OUTPUT_ROOT"])
    _write_run(output_root, "run-01")

    assert batch_client.get("/batches/run-01/summary.csv").mimetype == "text/csv"
    assert batch_client.get("/batches/run-01/results.json").mimetype == "application/json"
    assert batch_client.get("/batches/run-01/index.html").mimetype == "text/html"
    artifact = batch_client.get("/batches/run-01/artifacts/students/Alice_Example/diagnostics/beads/trace.txt")
    assert artifact.status_code == 200
    assert b"radius mismatch" in artifact.data
    escaped = batch_client.get("/batches/run-01/artifacts/../state.json")
    assert escaped.status_code in {400, 404}


def test_aggregate_leaderboard_uses_best_score_and_metric_direction(batch_client, tmp_path, monkeypatch):
    output_root = Path(os.environ["AUTOGRADER_BATCH_OUTPUT_ROOT"])
    run_a = _write_run(output_root, "run-a", homework_id="01")
    run_b = _write_run(output_root, "run-b", homework_id="01")
    result_b = json.loads((run_b / "results.json").read_text(encoding="utf-8"))
    result_b["students"][0]["problems"]["beads"]["points"] = 10
    result_b["students"][0]["metrics"]["beads"] = 1.0
    (run_b / "results.json").write_text(json.dumps(result_b), encoding="utf-8")
    state_b = json.loads((run_b / "state.json").read_text(encoding="utf-8"))
    state_b["finished_at"] = "2026-01-01T00:03:00+00:00"
    (run_b / "state.json").write_text(json.dumps(state_b), encoding="utf-8")

    response = batch_client.get("/batches/leaderboard?homework_id=01")

    assert response.status_code == 200
    assert b"Alice Example" in response.data
    assert b"10" in response.data
    assert b"1.0" in response.data
