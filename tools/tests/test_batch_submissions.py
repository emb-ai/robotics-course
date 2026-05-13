"""Tests for scanning local batch submissions."""

from zipfile import ZipFile

import pytest

from autograder.batch.submissions import scan_submissions


def test_scanner_collects_one_folder_per_student_nested_python_and_zip(tmp_path):
    root = tmp_path / "submissions"
    alice = root / "Alice"
    bob = root / "Bob"
    alice_nested = alice / "nested"
    alice_nested.mkdir(parents=True)
    bob.mkdir(parents=True)
    (alice_nested / "beads.py").write_text("answer = 'folder'\n")
    (alice / "notes.txt").write_text("ignore me\n")
    with ZipFile(bob / "submission.zip", "w") as zf:
        zf.writestr("deep/broom_racing.py", "answer = 'zip'\n")
        zf.writestr("readme.md", "ignore me\n")

    students = scan_submissions(root, ["beads.py", "broom_racing.py"])

    by_id = {student.student_id: student for student in students}
    assert by_id["Alice"].files == {"beads.py": "answer = 'folder'\n"}
    assert by_id["Bob"].files == {"broom_racing.py": "answer = 'zip'\n"}
    assert "notes.txt" in by_id["Alice"].ignored_files
    assert "submission.zip:readme.md" in by_id["Bob"].ignored_files


def test_scanner_marks_duplicate_solution_filename_as_error(tmp_path):
    root = tmp_path / "submissions"
    student = root / "Alice"
    (student / "nested").mkdir(parents=True)
    (student / "beads.py").write_text("a = 1\n")
    (student / "nested" / "beads.py").write_text("a = 2\n")

    [result] = scan_submissions(root, ["beads.py"])

    assert result.files == {}
    assert any("duplicate" in error.lower() for error in result.errors)


def test_scanner_rejects_unsafe_zip_entries(tmp_path):
    root = tmp_path / "submissions"
    student = root / "Alice"
    student.mkdir(parents=True)
    with ZipFile(student / "submission.zip", "w") as zf:
        zf.writestr("../beads.py", "bad = True\n")

    [result] = scan_submissions(root, ["beads.py"])

    assert result.files == {}
    assert any("unsafe" in error.lower() for error in result.errors)


def test_scanner_rejects_dataschool_download_root_in_local_mode(tmp_path):
    root = tmp_path / "dataschool-download"
    student_dir = root / "HW 1. Kinematics" / "Alice__100" / "timeline"
    student_dir.mkdir(parents=True)
    (root / "submissions.jsonl").write_text("{}\n", encoding="utf-8")
    (student_dir / "100-c001__solutions.zip").write_bytes(b"not a real zip")

    with pytest.raises(ValueError, match="DataSchool download root"):
        scan_submissions(root, ["beads.py"])
