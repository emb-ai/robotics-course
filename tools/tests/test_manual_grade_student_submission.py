"""Tests for the staff-only manual submission grading helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "dev" / "scripts" / "grade_student_submission.py"


def load_script():
    spec = importlib.util.spec_from_file_location("grade_student_submission", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_submission_files_filters_allowed_flat_python_files(tmp_path):
    module = load_script()
    (tmp_path / "beads.py").write_text("BEADS = 1\n", encoding="utf-8")
    (tmp_path / "broom_racing.py").write_text("BROOM = 2\n", encoding="utf-8")
    (tmp_path / "log.txt").write_text("old log\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "so101_ik.py").write_text("ignored\n", encoding="utf-8")

    files = module.collect_submission_files(
        tmp_path,
        ["beads.py", "broom_racing.py", "so101_ik.py"],
    )

    assert files == {
        "beads.py": "BEADS = 1\n",
        "broom_racing.py": "BROOM = 2\n",
    }


def test_collect_submission_files_errors_when_no_valid_solution_files(tmp_path):
    module = load_script()
    (tmp_path / "notes.py").write_text("wrong file\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid solution files"):
        module.collect_submission_files(tmp_path, ["beads.py"])


def test_mirror_submission_tree_copies_package_dir_and_skips_junk(tmp_path):
    module = load_script()
    submission_dir = tmp_path / "student"
    submission_dir.mkdir()
    (submission_dir / "beads.py").write_text("from .bead_chain import x\n", encoding="utf-8")
    (submission_dir / "log.txt").write_text("old log\n", encoding="utf-8")
    package_dir = submission_dir / "bead_chain"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "defaults.py").write_text("X = 1\n", encoding="utf-8")
    (package_dir / "beads_ppo_2_short.pt").write_bytes(b"\x00\x01checkpoint")
    cache_dir = package_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "defaults.cpython-310.pyc").write_bytes(b"pyc")

    dest = tmp_path / "mount"
    module.mirror_submission_tree(submission_dir, dest, ["beads.py", "broom_racing.py"])

    assert (dest / "__init__.py").is_file()
    assert (dest / "beads.py").read_text(encoding="utf-8") == "from .bead_chain import x\n"
    assert (dest / "bead_chain" / "defaults.py").read_text(encoding="utf-8") == "X = 1\n"
    assert (dest / "bead_chain" / "beads_ppo_2_short.pt").read_bytes() == b"\x00\x01checkpoint"
    assert not (dest / "log.txt").exists()
    assert not (dest / "bead_chain" / "__pycache__").exists()


def test_grade_submission_selects_tests_runs_docker_and_writes_log(tmp_path, monkeypatch):
    module = load_script()
    submission_dir = tmp_path / "student"
    submission_dir.mkdir()
    (submission_dir / "beads.py").write_text("answer = 42\n", encoding="utf-8")
    (submission_dir / "log.txt").write_text("stale log\n", encoding="utf-8")
    log_file = submission_dir / "log.txt"
    captured = {}

    def fake_get_solution_files(week_id):
        assert week_id == "01"
        return ["beads.py", "broom_racing.py"]

    def fake_get_test_paths_for_submission(week_id, files):
        captured["selected_files"] = dict(files)
        return ["tests/test_beads.py"]

    def fake_run(week_id, files, prepared_mount_dir=None):
        captured["run_week"] = week_id
        captured["run_files"] = dict(files)
        captured["prepared_mount_dir"] = prepared_mount_dir
        if prepared_mount_dir is not None:
            assert (prepared_mount_dir / "beads.py").read_text(encoding="utf-8") == "answer = 42\n"
            assert (prepared_mount_dir / "__init__.py").is_file()
        return 1, "STDOUT\n", "STDERR\n"

    monkeypatch.setattr(module, "get_solution_files", fake_get_solution_files)
    monkeypatch.setattr(module, "get_test_paths_for_submission", fake_get_test_paths_for_submission)
    monkeypatch.setattr(module, "docker_run", fake_run)

    result = module.grade_submission(
        week_id="01",
        submission_dir=submission_dir,
        log_file=log_file,
        skip_build=True,
    )

    assert result.submitted_files == ["beads.py"]
    assert result.selected_tests == ["tests/test_beads.py"]
    assert result.exit_code == 1
    assert captured["selected_files"] == {"beads.py": "answer = 42\n"}
    assert captured["run_week"] == "01"
    assert captured["run_files"] == {"beads.py": "answer = 42\n"}
    assert captured["prepared_mount_dir"] is not None
    log = log_file.read_text(encoding="utf-8")
    assert "submitted_files: beads.py" in log
    assert "selected_tests: tests/test_beads.py" in log
    assert "exit_code: 1" in log
    assert "STDOUT" in log
    assert "STDERR" in log
