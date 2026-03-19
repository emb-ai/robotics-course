"""Allowlist CRUD."""

from .schema import get_connection


def is_allowed(telegram_id: int) -> bool:
    """Check if user is in allowlist. If allowlist is empty, allow all (for initial setup)."""
    import os

    env_ids = os.environ.get("ALLOWED_TELEGRAM_IDS", "").strip()
    if env_ids:
        try:
            return telegram_id in [int(x.strip()) for x in env_ids.split(",") if x.strip()]
        except ValueError:
            pass

    conn = get_connection()
    try:
        cur = conn.execute("SELECT 1 FROM allowed_users LIMIT 1")
        if cur.fetchone() is None:
            return True  # Empty allowlist = allow all
        cur = conn.execute("SELECT 1 FROM allowed_users WHERE telegram_id = ?", (telegram_id,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def add_user(telegram_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("INSERT OR IGNORE INTO allowed_users (telegram_id) VALUES (?)", (telegram_id,))
        conn.commit()
    finally:
        conn.close()


def remove_user(telegram_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM allowed_users WHERE telegram_id = ?", (telegram_id,))
        conn.commit()
    finally:
        conn.close()


def list_users() -> list[int]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT telegram_id FROM allowed_users ORDER BY telegram_id").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()
