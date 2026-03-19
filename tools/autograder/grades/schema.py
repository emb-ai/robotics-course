"""SQLite schema for grades and students."""

import sqlite3
from pathlib import Path

_schema_initialized: set[str] = set()


def get_db_path() -> Path:
    import os

    base = os.environ.get("GRADES_DB_PATH")
    if base:
        return Path(base)
    # tools/autograder/grades/ -> tools/data/
    return Path(__file__).resolve().parent.parent.parent / "data" / "grades.db"


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            telegram_id INTEGER PRIMARY KEY,
            first_name TEXT,
            username TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL,
            week_id TEXT NOT NULL,
            problem_id TEXT NOT NULL,
            passed INTEGER NOT NULL,
            points INTEGER,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(telegram_id, week_id, problem_id)
        );

        CREATE INDEX IF NOT EXISTS idx_grades_telegram ON grades(telegram_id);
        CREATE INDEX IF NOT EXISTS idx_grades_week ON grades(week_id);

        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL,
            week_id TEXT NOT NULL,
            problem_id TEXT NOT NULL,
            metric_value REAL NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(telegram_id, week_id, problem_id)
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_week_problem ON metrics(week_id, problem_id);
    """)
    # Migration: add points column if missing (existing DBs)
    try:
        conn.execute("SELECT points FROM grades LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE grades ADD COLUMN points INTEGER")


def get_connection() -> sqlite3.Connection:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    path_str = str(path)
    if path_str not in _schema_initialized:
        init_schema(conn)
        _schema_initialized.add(path_str)
    return conn
