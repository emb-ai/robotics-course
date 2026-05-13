"""Tests for dashboard-launched batch job orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from autograder.batch import job_runner


def _write_job(tmp_path: Path, **overrides) -> Path:
    output_root = tmp_path / "batches"
    payload = {
        "run_id": "run-01",
        "homework_id": "01",
        "output_root": str(output_root),
        "max_workers": 3,
        "enable_diagnostics": True,
        "enable_feedback": False,
        "source_mode": "local",
        "submissions_root": str(tmp_path / "submissions"),
    }
    payload.update(overrides)
    path = output_root / payload["run_id"] / "job.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dataschool_downloader_module():
    path = Path(__file__).resolve().parents[2] / "dev" / "scripts" / "download_dataschool_submissions.py"
    spec = importlib.util.spec_from_file_location("download_dataschool_submissions_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_mode_calls_run_batch_with_configured_options(tmp_path, monkeypatch):
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    job_path = _write_job(tmp_path, submissions_root=str(submissions))
    calls = []

    def fake_run_batch(**kwargs):
        calls.append(kwargs)
        return {"students": []}

    monkeypatch.setattr(job_runner, "run_batch", fake_run_batch)

    assert job_runner.main(["--job-config", str(job_path)]) == 0

    assert calls == [
        {
            "homework_id": "01",
            "submissions_root": submissions,
            "output_root": tmp_path / "batches",
            "run_id": "run-01",
            "max_workers": 3,
            "enable_diagnostics": True,
            "enable_feedback": False,
        }
    ]


def test_dataschool_mode_downloads_and_prepares_latest_attempt(tmp_path, monkeypatch):
    download_root = tmp_path / "downloads"
    old_file = download_root / "assignment" / "Alice__100" / "timeline" / "old__beads.py"
    new_file = download_root / "assignment" / "Alice__100" / "timeline" / "new__beads.py"
    zip_file = download_root / "assignment" / "Alice__100" / "timeline" / "new__submission.zip"
    old_file.parent.mkdir(parents=True)
    old_file.write_text("old = True\n", encoding="utf-8")
    new_file.write_text("new = True\n", encoding="utf-8")
    with ZipFile(zip_file, "w") as zf:
        zf.writestr("broom_racing.py", "broom = True\n")
    interactions = [
        {
            "submission_id": "100",
            "student": "Alice Example",
            "assignment": "HW1",
            "gradeable_attempts": [
                {
                    "attempt_index": 1,
                    "created_at": "01.01.2026 10:00",
                    "role": "student",
                    "target": str(old_file),
                    "filename": "beads.py",
                },
                {
                    "attempt_index": 2,
                    "created_at": "02.01.2026 10:00",
                    "role": "student",
                    "target": str(new_file),
                    "filename": "beads.py",
                },
                {
                    "attempt_index": 2,
                    "created_at": "02.01.2026 10:00",
                    "role": "student",
                    "target": str(zip_file),
                    "filename": "submission.zip",
                },
            ],
        }
    ]
    (download_root / "submissions.jsonl").write_text(
        "\n".join(json.dumps(item) for item in interactions) + "\n",
        encoding="utf-8",
    )
    job_path = _write_job(
        tmp_path,
        source_mode="dataschool",
        download_root=str(download_root),
        queue_url="https://lk.dataschool.yandex.ru/teaching/assignments/?course=1704",
        cookie_file=str(tmp_path / "cookie.txt"),
    )
    (tmp_path / "cookie.txt").write_text("sessionid=secret", encoding="utf-8")
    subprocess_calls = []
    run_batch_calls = []

    class Completed:
        returncode = 0
        stdout = "downloaded"
        stderr = ""

    def fake_run(cmd, **kwargs):
        subprocess_calls.append((cmd, kwargs))
        return Completed()

    def fake_run_batch(**kwargs):
        run_batch_calls.append(kwargs)
        return {"students": []}

    monkeypatch.setattr(job_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(job_runner, "run_batch", fake_run_batch)

    assert job_runner.main(["--job-config", str(job_path)]) == 0

    assert subprocess_calls
    cmd = subprocess_calls[0][0]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("download_dataschool_submissions.py")
    assert cmd[2] == "--queue-url"
    assert "--cookie-file" in cmd
    assert "sessionid=secret" not in json.loads(job_path.read_text(encoding="utf-8")).values()
    prepared_root = tmp_path / "batches" / "run-01" / "prepared_submissions"
    prepared_student = prepared_root / "Alice_Example__100"
    assert (prepared_student / "beads.py").read_text(encoding="utf-8") == "new = True\n"
    assert (prepared_student / "broom_racing.py").read_text(encoding="utf-8") == "broom = True\n"
    manifest = json.loads((tmp_path / "batches" / "run-01" / "prepared_submissions.json").read_text())
    assert manifest["students"][0]["student_id"] == "Alice_Example__100"
    assert manifest["students"][0]["attempt_index"] == 2
    assert manifest["students"][0]["file_sources"]["broom_racing.py"]["archive_entry"] == "broom_racing.py"
    assert run_batch_calls[0]["submissions_root"] == prepared_root


def test_downloader_uses_student_name_folder_and_preserves_real_underscores(tmp_path):
    downloader = _dataschool_downloader_module()

    assert downloader.canonical_filename("broom_racing.py") == "broom_racing.py"
    assert downloader.canonical_filename("so101_ik_zlYepV5.py") == "so101_ik.py"

    target = downloader.stable_attachment_path(
        {
            "assignment": "HW 1. Kinematics",
            "student": "Александр Лазарев",
            "submission_id": "434293",
            "filename": "so101_ik.py",
            "event_id": "434293-c001",
        },
        tmp_path,
    )

    assert target == tmp_path / "HW 1. Kinematics" / "Александр Лазарев" / "timeline" / "434293-c001__so101_ik.py"


def test_prepare_dataschool_submissions_uses_latest_file_per_expected_solution(tmp_path):
    download_root = tmp_path / "downloads"
    timeline = download_root / "HW 1. Kinematics" / "Александр Лазарев__434293" / "timeline"
    timeline.mkdir(parents=True)
    files = {
        "434293-c001__so101_ik.py": "so101 = 'old'\n",
        "434293-c003__so101_ik_zlYepV5.py": "so101 = 'new'\n",
        "434293-c005__broom_racing.py": "broom = 'old'\n",
        "434293-c006__beads.py": "beads = 'only'\n",
        "434293-c008__broom_racing_7AYu8hx.py": "broom = 'new'\n",
    }
    for name, content in files.items():
        (timeline / name).write_text(content, encoding="utf-8")
    interaction = {
        "submission_id": "434293",
        "student": "Александр Лазарев",
        "assignment": "HW 1. Kinematics",
        "gradeable_attempts": [
            {
                "attempt_index": 1,
                "comment_index": 1,
                "created_at": "23.03.2026 02:24",
                "role": "student",
                "target": str(timeline / "434293-c001__so101_ik.py"),
                "filename": "so101_ik.py",
                "canonical_filename": "so101_ik.py",
                "attachment_id": "434293-c001-a01",
            },
            {
                "attempt_index": 2,
                "comment_index": 3,
                "created_at": "25.03.2026 18:26",
                "role": "student",
                "target": str(timeline / "434293-c003__so101_ik_zlYepV5.py"),
                "filename": "so101_ik_zlYepV5.py",
                "canonical_filename": "so101_ik.py",
                "attachment_id": "434293-c003-a01",
            },
            {
                "attempt_index": 1,
                "comment_index": 5,
                "created_at": "31.03.2026 01:14",
                "role": "student",
                "target": str(timeline / "434293-c005__broom_racing.py"),
                "filename": "broom_racing.py",
                "canonical_filename": "broom.py",
                "attachment_id": "434293-c005-a01",
            },
            {
                "attempt_index": 1,
                "comment_index": 6,
                "created_at": "31.03.2026 20:10",
                "role": "student",
                "target": str(timeline / "434293-c006__beads.py"),
                "filename": "beads.py",
                "canonical_filename": "beads.py",
                "attachment_id": "434293-c006-a01",
            },
            {
                "attempt_index": 1,
                "comment_index": 8,
                "created_at": "01.04.2026 23:55",
                "role": "student",
                "target": str(timeline / "434293-c008__broom_racing_7AYu8hx.py"),
                "filename": "broom_racing_7AYu8hx.py",
                "canonical_filename": "broom_racing.py",
                "attachment_id": "434293-c008-a01",
            },
        ],
    }
    (download_root / "submissions.jsonl").write_text(json.dumps(interaction, ensure_ascii=False) + "\n", encoding="utf-8")

    prepared = job_runner.prepare_dataschool_submissions(download_root, tmp_path / "run", homework_id="01")

    student_dir = prepared / "Александр_Лазарев__434293"
    assert sorted(path.name for path in student_dir.iterdir()) == ["beads.py", "broom_racing.py", "so101_ik.py"]
    assert (student_dir / "so101_ik.py").read_text(encoding="utf-8") == "so101 = 'new'\n"
    assert (student_dir / "broom_racing.py").read_text(encoding="utf-8") == "broom = 'new'\n"
    assert (student_dir / "beads.py").read_text(encoding="utf-8") == "beads = 'only'\n"
    manifest = json.loads((tmp_path / "run" / "prepared_submissions.json").read_text(encoding="utf-8"))
    assert manifest["students"][0]["files"] == ["beads.py", "broom_racing.py", "so101_ik.py"]
    assert manifest["students"][0]["file_sources"]["broom_racing.py"]["attachment_id"] == "434293-c008-a01"


def test_prepare_dataschool_submissions_extracts_latest_files_from_zip_per_solution(tmp_path):
    download_root = tmp_path / "downloads"
    timeline = download_root / "HW 1. Kinematics" / "Alice__100" / "timeline"
    timeline.mkdir(parents=True)
    old_so101 = timeline / "100-c001__so101_ik.py"
    old_beads = timeline / "100-c002__beads.py"
    zip_path = timeline / "100-c003__submission.zip"
    old_so101.write_text("so101 = 'old py'\n", encoding="utf-8")
    old_beads.write_text("beads = 'old py'\n", encoding="utf-8")
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("nested/so101_ik.py", "so101 = 'new zip'\n")
    interaction = {
        "submission_id": "100",
        "student": "Alice",
        "assignment": "HW1",
        "gradeable_attempts": [
            {
                "attempt_index": 1,
                "comment_index": 1,
                "role": "student",
                "target": str(old_so101),
                "filename": "so101_ik.py",
            },
            {
                "attempt_index": 1,
                "comment_index": 2,
                "role": "student",
                "target": str(old_beads),
                "filename": "beads.py",
            },
            {
                "attempt_index": 2,
                "comment_index": 3,
                "role": "student",
                "target": str(zip_path),
                "filename": "submission.zip",
                "attachment_id": "100-c003-a01",
            },
        ],
    }
    (download_root / "submissions.jsonl").write_text(json.dumps(interaction) + "\n", encoding="utf-8")

    prepared = job_runner.prepare_dataschool_submissions(download_root, tmp_path / "run", homework_id="01")

    student_dir = prepared / "Alice__100"
    assert sorted(path.name for path in student_dir.iterdir()) == ["beads.py", "so101_ik.py"]
    assert (student_dir / "so101_ik.py").read_text(encoding="utf-8") == "so101 = 'new zip'\n"
    assert (student_dir / "beads.py").read_text(encoding="utf-8") == "beads = 'old py'\n"
    manifest = json.loads((tmp_path / "run" / "prepared_submissions.json").read_text(encoding="utf-8"))
    assert manifest["students"][0]["file_sources"]["so101_ik.py"]["archive_entry"] == "nested/so101_ik.py"


def test_prepare_dataschool_submissions_preserves_non_latin_student_names(tmp_path):
    download_root = tmp_path / "downloads"
    submitted_file = download_root / "assignment" / "entry" / "timeline" / "so101_ik.py"
    submitted_file.parent.mkdir(parents=True)
    submitted_file.write_text("answer = 1\n", encoding="utf-8")
    (download_root / "submissions.jsonl").write_text(
        json.dumps(
            {
                "submission_id": "434413",
                "student": "Амир Ахундзянов",
                "assignment": "HW1",
                "gradeable_attempts": [
                    {
                        "attempt_index": 1,
                        "created_at": "29.03.2026 22:10",
                        "role": "student",
                        "target": str(submitted_file),
                        "filename": "so101_ik.py",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = job_runner.prepare_dataschool_submissions(download_root, tmp_path / "run")

    student_dir = prepared / "Амир_Ахундзянов__434413"
    assert (student_dir / "so101_ik.py").read_text(encoding="utf-8") == "answer = 1\n"
    manifest = json.loads((tmp_path / "run" / "prepared_submissions.json").read_text(encoding="utf-8"))
    assert manifest["students"][0]["student_id"] == "Амир_Ахундзянов__434413"
    assert manifest["students"][0]["student"] == "Амир Ахундзянов"


def test_prepare_dataschool_submissions_filters_to_selected_homework(tmp_path):
    download_root = tmp_path / "downloads"
    hw1_file = download_root / "HW 1. Kinematics" / "Alice__100" / "timeline" / "beads.py"
    hw2_file = download_root / "HW 2. Dynamics" / "Bob__200" / "timeline" / "solutions.zip"
    hw1_file.parent.mkdir(parents=True)
    hw2_file.parent.mkdir(parents=True)
    hw1_file.write_text("answer = 1\n", encoding="utf-8")
    hw2_file.write_bytes(b"zip-bytes")
    interactions = [
        {
            "submission_id": "100",
            "student": "Alice",
            "assignment": "HW 1. Kinematics",
            "gradeable_attempts": [
                {"attempt_index": 1, "role": "student", "target": str(hw1_file), "filename": "beads.py"}
            ],
        },
        {
            "submission_id": "200",
            "student": "Bob",
            "assignment": "HW 2. Dynamics",
            "gradeable_attempts": [
                {"attempt_index": 1, "role": "student", "target": str(hw2_file), "filename": "solutions.zip"}
            ],
        },
    ]
    (download_root / "submissions.jsonl").write_text(
        "\n".join(json.dumps(item) for item in interactions) + "\n",
        encoding="utf-8",
    )

    prepared = job_runner.prepare_dataschool_submissions(download_root, tmp_path / "run", homework_id="01")

    assert sorted(path.name for path in prepared.iterdir()) == ["Alice__100"]
    manifest = json.loads((tmp_path / "run" / "prepared_submissions.json").read_text(encoding="utf-8"))
    assert [student["assignment"] for student in manifest["students"]] == ["HW 1. Kinematics"]
    assert "skipped assignment" in manifest["warnings"][0]


def test_dataschool_downloader_failure_writes_error_state_and_skips_run_batch(tmp_path, monkeypatch):
    job_path = _write_job(
        tmp_path,
        source_mode="dataschool",
        download_root=str(tmp_path / "downloads"),
        queue_url="https://lk.dataschool.yandex.ru/teaching/assignments/?course=1704",
    )
    calls = []

    class Failed:
        returncode = 7
        stdout = "nope"
        stderr = "bad cookie"

    monkeypatch.setenv("DATASCHOOL_COOKIE", "sessionid=secret")
    monkeypatch.setattr(job_runner.subprocess, "run", lambda *args, **kwargs: Failed())
    monkeypatch.setattr(job_runner, "run_batch", lambda **kwargs: calls.append(kwargs))

    assert job_runner.main(["--job-config", str(job_path)]) == 7

    assert calls == []
    state = json.loads((tmp_path / "batches" / "run-01" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "error"
    assert "DataSchool download failed" in state["error"]


def test_prepare_dataschool_submissions_requires_gradeable_targets(tmp_path):
    run_dir = tmp_path / "run"
    download_root = tmp_path / "downloads"
    download_root.mkdir()
    (download_root / "submissions.jsonl").write_text(
        json.dumps({"student": "Alice", "submission_id": "1", "gradeable_attempts": []}) + "\n",
        encoding="utf-8",
    )

    prepared = job_runner.prepare_dataschool_submissions(download_root, run_dir)

    assert prepared == run_dir / "prepared_submissions"
    manifest = json.loads((run_dir / "prepared_submissions.json").read_text(encoding="utf-8"))
    assert manifest["students"] == []
    assert manifest["warnings"]
