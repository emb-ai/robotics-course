"""Tests for batch runner orchestration."""

import json
from pathlib import Path
from zipfile import ZipFile

import yaml

from autograder.batch import runner
from autograder.batch.runner import run_batch


def _write_homework(repo: Path) -> None:
    homework_dir = repo / "01-intro-and-kinematics" / "homework"
    container_dir = homework_dir / "container"
    container_dir.mkdir(parents=True)
    (container_dir / "docker_compose.yaml").write_text("services: {}\n")
    (homework_dir / "autograder.yaml").write_text(
        yaml.safe_dump(
            {
                "solution_files": ["beads.py", "broom_racing.py"],
                "problem_ids": {
                    "test_beads.py": "beads",
                    "test_broom_racing.py": "broom_racing",
                },
                "points": {"beads": 4, "broom_racing": 6},
                "limits": {"timeout_sec": 120, "memory_mb": 512, "cpus": 1, "network": "none"},
            }
        )
    )


def _write_submissions(root: Path) -> None:
    alice = root / "Alice"
    bob = root / "Bob"
    alice.mkdir(parents=True)
    bob.mkdir(parents=True)
    (alice / "beads.py").write_text("answer = 'ok'\n")
    with ZipFile(bob / "submission.zip", "w") as zf:
        zf.writestr("beads.py", "answer = 'bad'\n")


def test_run_batch_writes_artifacts_and_continues_after_failed_student(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submissions = tmp_path / "submissions"
    output = tmp_path / "batches"
    _write_homework(repo)
    _write_submissions(submissions)

    def fake_prebuild(spec, repo_root=None):
        return None

    def fake_run(spec, student_id, files, selected_tests, result_dir, repo_root=None, source_dir=None):
        pytest_xml = result_dir / "pytest.xml"
        if student_id == "Alice":
            pytest_xml.write_text(
                '<testsuite><testcase file="tests/test_beads.py" name="test_ok" /></testsuite>'
            )
            return {"exit_code": 0, "stdout": "METRIC:beads:1.5\n", "stderr": "", "elapsed_sec": 0.1}
        pytest_xml.write_text(
            '<testsuite><testcase file="tests/test_beads.py" name="test_bad">'
            '<failure message="bad">bad</failure></testcase></testsuite>'
        )
        return {"exit_code": 1, "stdout": "", "stderr": "failed", "elapsed_sec": 0.2}

    monkeypatch.setattr("autograder.batch.runner.prebuild_homework_image", fake_prebuild)
    monkeypatch.setattr("autograder.batch.runner.run_student_tests", fake_run)

    batch = run_batch("01", submissions, output, "run-01", max_workers=2, repo_root=repo)

    assert batch["homework_id"] == "01"
    assert {student["student_id"] for student in batch["students"]} == {"Alice", "Bob"}
    assert (output / "run-01" / "results.json").is_file()
    assert (output / "run-01" / "summary.csv").is_file()
    assert (output / "run-01" / "index.html").is_file()
    state = json.loads((output / "run-01" / "state.json").read_text())
    assert state["status"] == "done"
    by_id = {student["student_id"]: student for student in batch["students"]}
    assert by_id["Alice"]["problems"]["beads"]["status"] == "passed"
    assert by_id["Alice"]["metrics"] == {"beads": 1.5}
    assert by_id["Bob"]["problems"]["beads"]["status"] == "failed"


def test_run_batch_runs_selected_tests_individually_so_timeout_does_not_mask_other_problems(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submissions = tmp_path / "submissions"
    output = tmp_path / "batches"
    _write_homework(repo)
    _write_submissions(submissions)
    (submissions / "Alice" / "broom_racing.py").write_text("answer = 'ok'\n")
    calls = []

    def fake_prebuild(spec, repo_root=None):
        return None

    def fake_run(spec, student_id, files, selected_tests, result_dir, repo_root=None, source_dir=None):
        calls.append(list(selected_tests))
        if selected_tests == ["tests/test_beads.py"]:
            return {"exit_code": 143, "stdout": "collecting ...", "stderr": "", "elapsed_sec": 120.0}
        if selected_tests == ["tests/test_broom_racing.py"]:
            pytest_xml = result_dir / "pytest.xml"
            pytest_xml.write_text(
                '<testsuite><testcase file="tests/test_broom_racing.py" name="test_ok" /></testsuite>'
            )
            return {"exit_code": 0, "stdout": "broom ok", "stderr": "", "elapsed_sec": 0.1}
        raise AssertionError(f"expected one selected test, got {selected_tests!r}")

    monkeypatch.setattr("autograder.batch.runner.prebuild_homework_image", fake_prebuild)
    monkeypatch.setattr("autograder.batch.runner.run_student_tests", fake_run)

    batch = run_batch(
        "01",
        submissions,
        output,
        "run-01",
        max_workers=1,
        repo_root=repo,
        enable_diagnostics=False,
        enable_feedback=False,
    )

    by_id = {student["student_id"]: student for student in batch["students"]}
    assert calls[:2] == [["tests/test_beads.py"], ["tests/test_broom_racing.py"]]
    assert by_id["Alice"]["problems"]["beads"]["status"] == "timeout"
    assert by_id["Alice"]["problems"]["broom_racing"]["status"] == "passed"


def test_run_batch_adds_diagnostics_only_for_failed_students(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submissions = tmp_path / "submissions"
    output = tmp_path / "batches"
    _write_homework(repo)
    _write_submissions(submissions)
    (submissions / "Alice" / "broom_racing.py").write_text("answer = 'ok'\n")
    diagnosed = []
    feedback_calls = []

    def fake_prebuild(spec, repo_root=None):
        return None

    def fake_run(spec, student_id, files, selected_tests, result_dir, repo_root=None, source_dir=None):
        pytest_xml = result_dir / "pytest.xml"
        if student_id == "Alice":
            pytest_xml.write_text(
                '<testsuite><testcase file="tests/test_beads.py" name="test_ok" /></testsuite>'
            )
            return {"exit_code": 0, "stdout": "", "stderr": "", "elapsed_sec": 0.1}
        pytest_xml.write_text(
            '<testsuite><testcase file="tests/test_beads.py" name="test_bad">'
            '<failure message="bad">bad</failure></testcase></testsuite>'
        )
        return {"exit_code": 1, "stdout": "", "stderr": "failed", "elapsed_sec": 0.2}

    def fake_diagnostics(repo_root, spec, student_result, artifact_store):
        diagnosed.append(student_result["student_id"])
        return [
            {
                "plugin_id": "demo",
                "problem_id": "beads",
                "status": "ok",
                "summary": "diagnosed",
                "artifacts": [],
                "metrics": {},
                "error": None,
            }
        ]

    def fake_feedback(repo_root, spec, student_result, artifact_store):
        feedback_calls.append((student_result["student_id"], bool(student_result.get("diagnostics"))))
        student_result.setdefault("feedback", []).append({"problem_id": "beads", "status": "ok"})

    monkeypatch.setattr("autograder.batch.runner.prebuild_homework_image", fake_prebuild)
    monkeypatch.setattr("autograder.batch.runner.run_student_tests", fake_run)
    monkeypatch.setattr("autograder.batch.runner.run_diagnostics_for_student", fake_diagnostics)
    monkeypatch.setattr("autograder.batch.runner.generate_feedback_for_student", fake_feedback)

    batch = run_batch("01", submissions, output, "run-01", max_workers=2, repo_root=repo)

    by_id = {student["student_id"]: student for student in batch["students"]}
    assert diagnosed == ["Bob"]
    assert feedback_calls == [("Bob", True)]
    assert by_id["Alice"].get("diagnostics", []) == []
    assert by_id["Alice"].get("feedback", []) == []
    assert by_id["Bob"]["diagnostics"][0]["plugin_id"] == "demo"
    assert by_id["Bob"]["feedback"][0]["status"] == "ok"
    result_json = json.loads((output / "run-01" / "students" / "Bob" / "result.json").read_text())
    assert result_json["diagnostics"][0]["summary"] == "diagnosed"
    assert result_json["feedback"][0]["status"] == "ok"


def test_run_batch_continues_when_diagnostics_fail(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submissions = tmp_path / "submissions"
    output = tmp_path / "batches"
    _write_homework(repo)
    _write_submissions(submissions)

    def fake_prebuild(spec, repo_root=None):
        return None

    def fake_run(spec, student_id, files, selected_tests, result_dir, repo_root=None, source_dir=None):
        pytest_xml = result_dir / "pytest.xml"
        pytest_xml.write_text(
            '<testsuite><testcase file="tests/test_beads.py" name="test_bad">'
            '<failure message="bad">bad</failure></testcase></testsuite>'
        )
        return {"exit_code": 1, "stdout": "", "stderr": "failed", "elapsed_sec": 0.2}

    def broken_diagnostics(repo_root, spec, student_result, artifact_store):
        raise RuntimeError("diagnostics exploded")

    monkeypatch.setattr("autograder.batch.runner.prebuild_homework_image", fake_prebuild)
    monkeypatch.setattr("autograder.batch.runner.run_student_tests", fake_run)
    monkeypatch.setattr("autograder.batch.runner.run_diagnostics_for_student", broken_diagnostics)

    batch = run_batch("01", submissions, output, "run-01", max_workers=2, repo_root=repo)

    assert batch["students"]
    assert all(student["status"] == "failed" for student in batch["students"])


def test_run_batch_continues_when_feedback_fails(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submissions = tmp_path / "submissions"
    output = tmp_path / "batches"
    _write_homework(repo)
    _write_submissions(submissions)

    def fake_prebuild(spec, repo_root=None):
        return None

    def fake_run(spec, student_id, files, selected_tests, result_dir, repo_root=None, source_dir=None):
        pytest_xml = result_dir / "pytest.xml"
        pytest_xml.write_text(
            '<testsuite><testcase file="tests/test_beads.py" name="test_bad">'
            '<failure message="bad">bad</failure></testcase></testsuite>'
        )
        return {"exit_code": 1, "stdout": "", "stderr": "failed", "elapsed_sec": 0.2}

    def fake_diagnostics(repo_root, spec, student_result, artifact_store):
        return []

    def broken_feedback(repo_root, spec, student_result, artifact_store):
        raise RuntimeError("feedback exploded")

    monkeypatch.setattr("autograder.batch.runner.prebuild_homework_image", fake_prebuild)
    monkeypatch.setattr("autograder.batch.runner.run_student_tests", fake_run)
    monkeypatch.setattr("autograder.batch.runner.run_diagnostics_for_student", fake_diagnostics)
    monkeypatch.setattr("autograder.batch.runner.generate_feedback_for_student", broken_feedback)

    batch = run_batch("01", submissions, output, "run-01", max_workers=1, repo_root=repo)

    assert batch["students"]
    assert all(student["status"] == "failed" for student in batch["students"])
    assert all(student["feedback"] == [] for student in batch["students"])
    assert all(student["feedback_error"] == "feedback exploded" for student in batch["students"])
    result_json = json.loads((output / "run-01" / "students" / "Alice" / "result.json").read_text())
    assert result_json["feedback_error"] == "feedback exploded"


def test_run_batch_can_disable_diagnostics_and_feedback(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submissions = tmp_path / "submissions"
    output = tmp_path / "batches"
    _write_homework(repo)
    _write_submissions(submissions)

    def fake_prebuild(spec, repo_root=None):
        return None

    def fake_run(spec, student_id, files, selected_tests, result_dir, repo_root=None, source_dir=None):
        pytest_xml = result_dir / "pytest.xml"
        pytest_xml.write_text(
            '<testsuite><testcase file="tests/test_beads.py" name="test_bad">'
            '<failure message="bad">bad</failure></testcase></testsuite>'
        )
        return {"exit_code": 1, "stdout": "", "stderr": "failed", "elapsed_sec": 0.2}

    def unexpected_diagnostics(*args, **kwargs):
        raise AssertionError("diagnostics should be disabled")

    def unexpected_feedback(*args, **kwargs):
        raise AssertionError("feedback should be disabled")

    monkeypatch.setattr("autograder.batch.runner.prebuild_homework_image", fake_prebuild)
    monkeypatch.setattr("autograder.batch.runner.run_student_tests", fake_run)
    monkeypatch.setattr("autograder.batch.runner.run_diagnostics_for_student", unexpected_diagnostics)
    monkeypatch.setattr("autograder.batch.runner.generate_feedback_for_student", unexpected_feedback)

    batch = run_batch(
        "01",
        submissions,
        output,
        "run-01",
        max_workers=1,
        repo_root=repo,
        enable_diagnostics=False,
        enable_feedback=False,
    )

    assert batch["students"]
    assert all(student["diagnostics"] == [] for student in batch["students"])
    assert all(student["feedback"] == [] for student in batch["students"])


def test_batch_runner_cli_passes_disable_flags(monkeypatch):
    calls = []

    def fake_run_batch(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(runner, "run_batch", fake_run_batch)

    assert runner.main(
        [
            "--homework",
            "01",
            "--submissions-root",
            "/tmp/submissions",
            "--output-root",
            "/tmp/batches",
            "--run-id",
            "run-01",
            "--no-diagnostics",
            "--no-feedback",
        ]
    ) == 0

    assert calls
    assert calls[0][1]["enable_diagnostics"] is False
    assert calls[0][1]["enable_feedback"] is False
