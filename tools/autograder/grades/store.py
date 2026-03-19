"""Grade storage and leaderboard queries."""

import sqlite3

from .schema import get_connection


def _is_better(current: float, new_val: float, direction: str) -> bool:
    """Return True if new_val is better than current (given direction: minimize/maximize)."""
    if direction == "minimize":
        return new_val < current
    return new_val > current


def upsert_student(
    conn: sqlite3.Connection,
    telegram_id: int,
    first_name: str | None = None,
    username: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO students (telegram_id, first_name, username, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(telegram_id) DO UPDATE SET
            first_name = excluded.first_name,
            username = excluded.username,
            updated_at = CURRENT_TIMESTAMP
        """,
        (telegram_id, first_name or "", username or ""),
    )


def upsert_grade(
    conn: sqlite3.Connection,
    telegram_id: int,
    week_id: str,
    problem_id: str,
    passed: int,
    points: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO grades (telegram_id, week_id, problem_id, passed, points, submitted_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(telegram_id, week_id, problem_id) DO UPDATE SET
            passed = excluded.passed,
            points = excluded.points,
            submitted_at = CURRENT_TIMESTAMP
        """,
        (telegram_id, week_id, problem_id, passed, points),
    )


def upsert_grades_batch(
    conn: sqlite3.Connection,
    telegram_id: int,
    week_id: str,
    problem_results: dict[str, int],
    problem_points: dict[str, int] | None = None,
    first_name: str | None = None,
    username: str | None = None,
) -> None:
    upsert_student(conn, telegram_id, first_name, username)
    pts_cfg = problem_points or {}
    for problem_id, passed in problem_results.items():
        scored = (passed * pts_cfg.get(problem_id, 1)) if pts_cfg else (passed if passed else 0)
        upsert_grade(conn, telegram_id, week_id, problem_id, passed, scored)
    conn.commit()


def upsert_metric(
    conn: sqlite3.Connection,
    telegram_id: int,
    week_id: str,
    problem_id: str,
    metric_value: float,
    direction: str = "minimize",
) -> None:
    """
    Upsert metric. Keeps best value: min for minimize, max for maximize.
    """
    row = conn.execute(
        "SELECT metric_value FROM metrics WHERE telegram_id=? AND week_id=? AND problem_id=?",
        (telegram_id, week_id, problem_id),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO metrics (telegram_id, week_id, problem_id, metric_value, submitted_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(telegram_id, week_id, problem_id) DO UPDATE SET
                metric_value = excluded.metric_value,
                submitted_at = CURRENT_TIMESTAMP
            """,
            (telegram_id, week_id, problem_id, metric_value),
        )
    else:
        current = row[0]
        if _is_better(current, metric_value, direction):
            conn.execute(
                """
                UPDATE metrics SET metric_value=?, submitted_at=CURRENT_TIMESTAMP
                WHERE telegram_id=? AND week_id=? AND problem_id=?
                """,
                (metric_value, telegram_id, week_id, problem_id),
            )


def get_metric_leaderboard(
    week_id: str,
    problem_id: str,
    direction: str = "minimize",
    limit: int = 20,
) -> list[dict]:
    """Return leaderboard for a metric: [{telegram_id, first_name, username, display, metric_value}, ...]."""
    conn = get_connection()
    order = "ASC" if direction == "minimize" else "DESC"
    rows = conn.execute(
        f"""
        SELECT m.telegram_id, s.first_name, s.username, m.metric_value
        FROM metrics m
        JOIN students s ON m.telegram_id = s.telegram_id
        WHERE m.week_id = ? AND m.problem_id = ?
        ORDER BY m.metric_value {order}
        LIMIT ?
        """,
        (week_id, problem_id, limit),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        display = (r["username"] or r["first_name"] or str(r["telegram_id"])).strip() or str(r["telegram_id"])
        out.append({
            "telegram_id": r["telegram_id"],
            "first_name": r["first_name"],
            "username": r["username"],
            "display": display,
            "metric_value": r["metric_value"],
        })
    return out


def get_my_grades(telegram_id: int) -> list[dict]:
    """Return user's grades as list of {week_id, problem_id, passed, points}."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT week_id, problem_id, passed, points
        FROM grades
        WHERE telegram_id = ?
        ORDER BY week_id, problem_id
        """,
        (telegram_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_leaderboard() -> list[dict]:
    """Return all grades with student names: {telegram_id, first_name, username, week_id, problem_id, passed, points}."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT s.telegram_id, s.first_name, s.username, g.week_id, g.problem_id, g.passed, g.points
        FROM grades g
        JOIN students s ON g.telegram_id = s.telegram_id
        ORDER BY s.telegram_id, g.week_id, g.problem_id
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_leaderboard_pivoted() -> list[dict]:
    """Return leaderboard as pivot table: one row per student, columns = week_problem (points), total_pts."""
    raw = get_leaderboard()
    if not raw:
        return []

    by_user: dict[tuple, dict[str, str | int]] = {}
    for r in raw:
        key = (r["telegram_id"], r["first_name"] or "", r["username"] or "")
        if key not in by_user:
            by_user[key] = {
                "telegram_id": r["telegram_id"],
                "first_name": r["first_name"] or "",
                "username": r["username"] or "",
                "display": (r["username"] or r["first_name"] or str(r["telegram_id"])).strip() or str(r["telegram_id"]),
                "total_pts": 0,
            }
        col = f"{r['week_id']}_{r['problem_id']}"
        pts = r.get("points")
        if pts is not None:
            by_user[key][col] = pts
            by_user[key]["total_pts"] = by_user[key].get("total_pts", 0) + pts
        else:
            by_user[key][col] = "✓" if r["passed"] else "✗"

    return list(by_user.values())
