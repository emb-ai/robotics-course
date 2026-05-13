"""File-backed leaderboards for reports-only batch grading runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .discovery import get_homework


def list_runs(output_root: str | Path) -> list[dict[str, Any]]:
    """Return recent local batch runs with state/config/result metadata."""

    root = Path(output_root)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name, reverse=True):
        state = _read_json(run_dir / "state.json", {})
        config = _read_json(run_dir / "config.json", {})
        job = _read_json(run_dir / "job.json", {})
        results = _read_json(run_dir / "results.json", {})
        homework_id = results.get("homework_id") or config.get("homework_id") or job.get("homework_id") or ""
        runs.append(
            {
                "run_id": run_dir.name,
                "homework_id": homework_id,
                "status": state.get("status", "unknown"),
                "counts": state.get("counts", {}),
                "started_at": state.get("started_at") or state.get("created_at"),
                "finished_at": state.get("finished_at"),
                "has_results": bool(results),
            }
        )
    return runs


def per_run_leaderboards(results: dict[str, Any], repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build points and metric leaderboards for one batch result."""

    homework_id = str(results.get("homework_id") or "")
    metrics_config = _metrics_config(homework_id, repo_root)
    return {
        "points": sorted(
            [_student_points_row(student) for student in results.get("students", []) if isinstance(student, dict)],
            key=lambda row: (-row["total_points"], row["student_id"]),
        ),
        "metrics": _metric_boards(results.get("students", []), metrics_config),
    }


def aggregate_leaderboards(
    output_root: str | Path,
    repo_root: str | Path | None = None,
    homework_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate best local batch scores across finished runs."""

    root = Path(output_root)
    best_points: dict[tuple[str, str], dict[str, Any]] = {}
    best_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not root.exists():
        return {"points": [], "metrics": []}

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        state = _read_json(run_dir / "state.json", {})
        if state.get("status") != "done":
            continue
        results = _read_json(run_dir / "results.json", {})
        run_homework = str(results.get("homework_id") or "")
        if homework_id and run_homework != homework_id:
            continue
        finished_at = str(state.get("finished_at") or "")
        metrics_config = _metrics_config(run_homework, repo_root)
        for student in results.get("students", []):
            if not isinstance(student, dict):
                continue
            points_row = _student_points_row(student)
            points_row.update({"homework_id": run_homework, "run_id": run_dir.name, "finished_at": finished_at})
            points_key = (run_homework, points_row["student_id"])
            current = best_points.get(points_key)
            if current is None or _is_better_points(points_row, current):
                best_points[points_key] = points_row

            for problem_id, value in (student.get("metrics") or {}).items():
                direction = str(metrics_config.get(str(problem_id), {}).get("direction", "minimize"))
                metric_row = {
                    "homework_id": run_homework,
                    "problem_id": str(problem_id),
                    "name": metrics_config.get(str(problem_id), {}).get("name", str(problem_id)),
                    "direction": direction,
                    "student_id": points_row["student_id"],
                    "value": float(value),
                    "run_id": run_dir.name,
                    "finished_at": finished_at,
                }
                metric_key = (run_homework, str(problem_id), points_row["student_id"])
                current_metric = best_metrics.get(metric_key)
                if current_metric is None or _is_better_metric(metric_row, current_metric):
                    best_metrics[metric_key] = metric_row

    point_rows = sorted(best_points.values(), key=lambda row: (-row["total_points"], row["homework_id"], row["student_id"]))
    return {"points": point_rows, "metrics": _group_metric_rows(best_metrics.values())}


def _student_points_row(student: dict[str, Any]) -> dict[str, Any]:
    problems = student.get("problems") if isinstance(student.get("problems"), dict) else {}
    total = 0.0
    max_total = 0.0
    for problem in problems.values():
        if not isinstance(problem, dict):
            continue
        total += _float(problem.get("points"))
        max_total += _float(problem.get("max_points"))
    return {
        "student_id": str(student.get("student_id") or ""),
        "student_path_id": str(student.get("student_path_id") or student.get("student_id") or ""),
        "status": str(student.get("status") or ""),
        "total_points": total,
        "max_points": max_total,
    }


def _metric_boards(students: list[Any], metrics_config: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for student in students:
        if not isinstance(student, dict):
            continue
        student_id = str(student.get("student_id") or "")
        for problem_id, value in (student.get("metrics") or {}).items():
            problem = str(problem_id)
            cfg = metrics_config.get(problem, {})
            grouped.setdefault(problem, []).append(
                {
                    "student_id": student_id,
                    "value": float(value),
                    "problem_id": problem,
                    "name": cfg.get("name", problem),
                    "direction": cfg.get("direction", "minimize"),
                }
            )
    return _group_metric_rows(
        row
        for rows in grouped.values()
        for row in rows
    )


def _group_metric_rows(rows: Any) -> list[dict[str, Any]]:
    boards: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("homework_id", "")), str(row["problem_id"]))
        board = boards.setdefault(
            key,
            {
                "homework_id": row.get("homework_id", ""),
                "problem_id": row["problem_id"],
                "name": row.get("name", row["problem_id"]),
                "direction": row.get("direction", "minimize"),
                "rows": [],
            },
        )
        board["rows"].append(row)
    for board in boards.values():
        reverse = board["direction"] == "maximize"
        board["rows"].sort(key=lambda item: (item["value"], item["student_id"]), reverse=reverse)
    return sorted(boards.values(), key=lambda item: (item.get("homework_id", ""), item["problem_id"]))


def _is_better_points(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    if candidate["total_points"] != current["total_points"]:
        return candidate["total_points"] > current["total_points"]
    return str(candidate.get("finished_at") or "") > str(current.get("finished_at") or "")


def _is_better_metric(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    direction = candidate.get("direction", "minimize")
    if candidate["value"] != current["value"]:
        return candidate["value"] > current["value"] if direction == "maximize" else candidate["value"] < current["value"]
    return str(candidate.get("finished_at") or "") > str(current.get("finished_at") or "")


def _metrics_config(homework_id: str, repo_root: str | Path | None) -> dict[str, dict[str, Any]]:
    if not homework_id:
        return {}
    try:
        spec = get_homework(repo_root or Path(__file__).resolve().parents[3], homework_id)
    except Exception:
        return {}
    return {
        str(problem_id): cfg
        for problem_id, cfg in (spec.metrics or {}).items()
        if isinstance(cfg, dict)
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default

