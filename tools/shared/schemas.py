"""Shared schemas for job payload and grade rows."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Job:
    """Grading job payload (bot -> Redis -> autograder)."""

    chat_id: int
    week_id: str
    files: dict[str, str]  # filename -> content (UTF-8)
    user_id: int  # Telegram user ID
    first_name: str | None = None
    username: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "week_id": self.week_id,
            "files": self.files,
            "user_id": self.user_id,
            "first_name": self.first_name,
            "username": self.username,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        return cls(
            chat_id=int(d["chat_id"]),
            week_id=str(d["week_id"]),
            files=dict(d["files"]),
            user_id=int(d["user_id"]),
            first_name=d.get("first_name"),
            username=d.get("username"),
        )


@dataclass
class GradeRow:
    """One grade record (student, week, problem, passed)."""

    telegram_id: int
    week_id: str
    problem_id: str
    passed: int  # 0 or 1
