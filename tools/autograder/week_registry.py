"""Week registry: week_id -> compose path, topic slug, solution files, problem_ids."""

from pathlib import Path

from shared.week_config import (
    get_limits,
    get_metrics_config,
    get_points,
    get_problem_ids,
    get_repo_root,
    get_solution_files,
    get_topic_slug,
    get_week_config,
    list_weeks,
)


def get_compose_path(week_id: str) -> Path:
    cfg = get_week_config(week_id)
    return get_repo_root() / cfg["compose_file"]


def get_solutions_mount_path(week_id: str) -> str:
    """Container path where solutions are mounted, e.g. /app/01-intro-and-kinematics/homework/solutions."""
    slug = get_topic_slug(week_id)
    return f"/app/{slug}/homework/solutions"


def get_test_paths_for_submission(week_id: str, solution_filenames: dict[str, str] | None = None) -> list[str]:
    """
    Return test paths to run for the given submission (only tests for submitted solution files).
    Avoids importing solution modules that were not submitted. solution_filenames is the job.files
    dict (filename -> content); only keys are used. If None or empty, returns all test paths.
    """
    problem_ids = get_problem_ids(week_id)
    if not problem_ids:
        return []
    submitted = set(solution_filenames.keys()) if solution_filenames else None
    if not submitted:
        return [f"tests/{tf}" for tf in problem_ids]
    # problem_ids: test_file -> problem_id (solution file base); solution file = f"{problem_id}.py"
    return [
        f"tests/{test_file}"
        for test_file, pid in problem_ids.items()
        if f"{pid}.py" in submitted
    ]


__all__ = [
    "get_compose_path",
    "get_limits",
    "get_metrics_config",
    "get_points",
    "get_problem_ids",
    "get_repo_root",
    "get_solution_files",
    "get_solutions_mount_path",
    "get_test_paths_for_submission",
    "get_topic_slug",
    "list_weeks",
]
