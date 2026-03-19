"""Tests for autograder/grades/store.py."""

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def grades_db_path(tmp_path):
    """Temp SQLite DB for grades."""
    return tmp_path / "grades.db"


@pytest.fixture
def grades_conn(grades_db_path, monkeypatch):
    """Connection to temp grades DB with schema."""
    monkeypatch.setenv("GRADES_DB_PATH", str(grades_db_path))
    # Clear schema init cache so we get fresh schema
    import autograder.grades.schema as mod
    mod._schema_initialized.clear()
    conn = mod.get_connection()
    yield conn
    conn.close()


def test_upsert_student(grades_conn):
    from autograder.grades.store import upsert_student, upsert_grade

    upsert_student(grades_conn, 111, "Alice", "alice")
    upsert_grade(grades_conn, 111, "01", "beads", 1)
    grades_conn.commit()

    rows = grades_conn.execute("SELECT * FROM students WHERE telegram_id = 111").fetchall()
    assert len(rows) == 1
    assert rows[0]["first_name"] == "Alice"


def test_upsert_grades_batch(grades_conn):
    from autograder.grades.store import upsert_grades_batch, get_my_grades

    upsert_grades_batch(
        grades_conn, 222, "01",
        {"beads": 1, "broom_racing": 0, "so101_ik": 1},
        first_name="Bob", username="bob",
    )

    rows = get_my_grades(222)
    assert len(rows) == 3
    passed = {r["problem_id"]: r["passed"] for r in rows}
    assert passed["beads"] == 1
    assert passed["broom_racing"] == 0
    assert passed["so101_ik"] == 1


def test_upsert_overwrites(grades_conn):
    from autograder.grades.store import upsert_grades_batch, get_my_grades

    upsert_grades_batch(grades_conn, 333, "01", {"beads": 0}, first_name="Charlie", username="charlie")
    upsert_grades_batch(grades_conn, 333, "01", {"beads": 1}, first_name="Charlie", username="charlie")

    rows = get_my_grades(333)
    assert len(rows) == 1
    assert rows[0]["passed"] == 1


def test_get_leaderboard_pivoted(grades_conn):
    from autograder.grades.store import upsert_grades_batch, get_leaderboard_pivoted

    upsert_grades_batch(grades_conn, 1, "01", {"beads": 1, "broom_racing": 0}, first_name="A", username="a")
    upsert_grades_batch(grades_conn, 2, "01", {"beads": 1, "broom_racing": 1}, first_name="B", username="b")

    rows = get_leaderboard_pivoted()
    assert len(rows) == 2
    display_vals = [r["display"] for r in rows]
    assert "A" in display_vals or "a" in display_vals
    assert "B" in display_vals or "b" in display_vals


def test_get_my_grades_empty(grades_conn):
    from autograder.grades.store import get_my_grades

    rows = get_my_grades(999)
    assert rows == []
