"""Tests for batch grading artifact storage."""

import json
from pathlib import Path

import pytest

from autograder.batch.artifacts import BatchArtifactStore


def _sample_batch_result() -> dict:
    return {
        "problem_ids": ["beads", "broom"],
        "points": {"beads": 4, "broom": 6},
        "students": [
            {
                "student_id": "Alice Example",
                "problems": {
                    "beads": {
                        "status": "passed",
                        "points": 4,
                        "max_points": 4,
                    }
                },
            },
            {
                "student_id": "bob",
                "problems": {
                    "beads": {
                        "status": "failed",
                        "points": 0,
                        "max_points": 4,
                    },
                    "broom": {
                        "status": "passed",
                        "points": 6,
                        "max_points": 6,
                    },
                },
            },
        ],
    }


def test_initializes_run_directory_and_state(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")
    store.write_config({"week_id": "01", "workers": 2})

    assert store.run_dir == temp_dir / "run-01"
    assert (store.run_dir / "students").is_dir()
    assert json.loads((store.run_dir / "config.json").read_text()) == {
        "week_id": "01",
        "workers": 2,
    }
    assert json.loads((store.run_dir / "state.json").read_text())["status"] == "created"


@pytest.mark.parametrize("run_id", ["../evil", "nested/run", "/tmp/run", "", ".", ".."])
def test_rejects_unsafe_run_id(temp_dir, run_id):
    with pytest.raises(ValueError):
        BatchArtifactStore(temp_dir, run_id)


@pytest.mark.parametrize("student_id", ["../evil", "nested/student", "", ".", ".."])
def test_rejects_unsafe_student_id(temp_dir, student_id):
    store = BatchArtifactStore(temp_dir, "run-01")

    with pytest.raises(ValueError):
        store.student_dir(student_id)


def test_sanitizes_student_path_and_preserves_display_id(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")
    student_dir = store.student_dir("Alice Example")

    assert student_dir.name.startswith("Alice_Example")
    metadata = json.loads((student_dir / "student.json").read_text())
    assert metadata["student_id"] == "Alice Example"
    assert metadata["student_path_id"] == student_dir.name


def test_atomic_json_write_leaves_valid_json_after_repeated_writes(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")

    for i in range(40):
        store.update_state({"status": "running", "completed": i})
        loaded = json.loads((store.run_dir / "state.json").read_text())
        assert loaded["completed"] == i

    leftovers = list(store.run_dir.glob(".state.json.*.tmp"))
    assert leftovers == []


def test_artifact_ref_must_stay_inside_run_dir(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")
    inside = store.write_text_artifact(
        "Alice",
        "beads",
        "diagnostic",
        "trace.txt",
        "radius exploded",
    )

    assert inside["path"] == "students/Alice/diagnostics/beads/trace.txt"
    with pytest.raises(ValueError):
        store.register_artifact_ref(
            "Alice",
            "beads",
            "diagnostic",
            "outside",
            temp_dir / "outside.txt",
        )


def test_summary_csv_has_one_row_per_student_problem_including_missing(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")
    store.write_summary_csv(_sample_batch_result())

    rows = (store.run_dir / "summary.csv").read_text().splitlines()
    assert rows[0] == "student_id,student_path_id,problem_id,status,points,max_points"
    assert "Alice Example,Alice_Example,beads,passed,4,4" in rows
    assert "Alice Example,Alice_Example,broom,missing,0,6" in rows
    assert "bob,bob,broom,passed,6,6" in rows
    assert len(rows) == 5


def test_index_html_includes_students_statuses_and_artifact_links(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")
    store.write_text_artifact(
        "bob",
        "beads",
        "diagnostic",
        "plot.html",
        "<p>diagnostic</p>",
    )
    store.write_feedback(
        "bob",
        "beads",
        "Check the radius update.",
        {"severity": "medium"},
    )

    store.write_index_html(_sample_batch_result())
    html = (store.run_dir / "index.html").read_text()

    assert "Alice Example" in html
    assert "bob" in html
    assert "failed" in html
    assert "students/bob/diagnostics/beads/plot.html" in html
    assert "students/bob/feedback/beads.md" in html
    assert "students/bob/feedback/beads.json" in html


def test_submitted_file_copy_preserves_content_and_cannot_escape(temp_dir):
    source_root = temp_dir / "source"
    source_root.mkdir()
    source = source_root / "beads.py"
    source.write_text("answer = 42\n")
    store = BatchArtifactStore(temp_dir / "out", "run-01")

    copied = store.copy_submitted_file("Alice", "beads.py", source)
    assert copied.read_text() == "answer = 42\n"
    assert copied == store.run_dir / "students" / "Alice" / "submitted" / "beads.py"

    with pytest.raises(ValueError):
        store.copy_submitted_file("Alice", "../evil.py", source)

    assert not (store.run_dir / "students" / "evil.py").exists()


def test_student_result_uses_display_id_and_relative_artifact_refs(temp_dir):
    store = BatchArtifactStore(temp_dir, "run-01")
    ref = store.write_bytes_artifact(
        "Alice Example",
        "beads",
        "diagnostic",
        "trace.bin",
        b"\x00\x01",
    )
    store.write_student_result(
        "Alice Example",
        {
            "status": "failed",
            "artifacts": [ref],
        },
    )

    result = json.loads(
        (store.student_dir("Alice Example") / "result.json").read_text()
    )
    assert result["student_id"] == "Alice Example"
    assert result["student_path_id"] == "Alice_Example"
    assert result["artifacts"][0]["path"] == (
        "students/Alice_Example/diagnostics/beads/trace.bin"
    )
