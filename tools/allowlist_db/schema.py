"""Allowlist SQLite schema."""

import os
import sqlite3
from pathlib import Path

_schema_initialized: set[str] = set()


def get_db_path() -> Path:
    base = os.environ.get("ALLOWLIST_DB_PATH")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data" / "allowlist.db"


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS allowed_users (
            telegram_id INTEGER PRIMARY KEY
        )
    """)


def get_connection() -> sqlite3.Connection:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    path_str = str(path)
    if path_str not in _schema_initialized:
        init_schema(conn)
        _schema_initialized.add(path_str)
    return conn
