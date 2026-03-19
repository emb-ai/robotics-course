"""Tests for shared/schemas.py."""

import pytest

from shared.schemas import Job, GradeRow


def test_job_to_dict_roundtrip():
    job = Job(
        chat_id=123,
        week_id="01",
        files={"beads.py": "print(1)"},
        user_id=456,
        first_name="Alice",
        username="alice",
    )
    d = job.to_dict()
    restored = Job.from_dict(d)
    assert restored.chat_id == job.chat_id
    assert restored.week_id == job.week_id
    assert restored.files == job.files
    assert restored.user_id == job.user_id
    assert restored.first_name == job.first_name
    assert restored.username == job.username


def test_job_from_dict_missing_optionals():
    d = {
        "chat_id": 1,
        "week_id": "01",
        "files": {},
        "user_id": 2,
    }
    job = Job.from_dict(d)
    assert job.first_name is None
    assert job.username is None


def test_job_from_dict_with_optionals():
    d = {
        "chat_id": 1,
        "week_id": "01",
        "files": {"a.py": "x"},
        "user_id": 2,
        "first_name": "Bob",
        "username": "bob",
    }
    job = Job.from_dict(d)
    assert job.first_name == "Bob"
    assert job.username == "bob"


def test_grade_row_dataclass():
    row = GradeRow(telegram_id=1, week_id="01", problem_id="beads", passed=1)
    assert row.telegram_id == 1
    assert row.week_id == "01"
    assert row.problem_id == "beads"
    assert row.passed == 1
