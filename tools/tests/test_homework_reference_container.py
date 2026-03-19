"""Assert reference code in dev/ passes open + hidden tests in the homework container.

The container copies reference_solution/*.py into solutions/ unless GRADING_STUDENT_SUBMISSION=1
(autograder). run.sh does not set the latter, so this matches `homework/container/run.sh`.
"""

import os
import subprocess
from pathlib import Path

import pytest

from shared.week_config import (
    get_solution_files,
    get_topic_slug,
    list_weeks_with_homework_container,
)

# Repo root = parent of tools/ (same as conftest repo_root when tests live under tools/tests).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Cold Docker build + full pytest can take several minutes.
_CONTAINER_TIMEOUT_SEC = int(os.environ.get("HOMEWORK_REF_CONTAINER_TIMEOUT", "1200"))


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=30,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _missing_solution_files(repo: Path, week_id: str) -> list[str]:
    """Filenames from autograder ``solution_files`` missing under dev/.../reference_solution/."""
    topic_slug = get_topic_slug(week_id)
    ref = repo / "dev" / topic_slug / "homework" / "reference_solution"
    missing: list[str] = []
    for name in get_solution_files(week_id):
        if not (ref / name).is_file():
            missing.append(name)
    return missing


@pytest.fixture(scope="module")
def docker_ok():
    if not _docker_available():
        pytest.skip("Docker not available (docker info failed)")
    return True


@pytest.mark.integration
@pytest.mark.parametrize("week_id", list_weeks_with_homework_container(_REPO_ROOT))
def test_reference_solution_passes_in_homework_container(week_id: str, docker_ok, repo_root):
    """Full grading run: open tests + hidden_tests + reference comparisons."""
    repo = Path(repo_root) if not isinstance(repo_root, Path) else repo_root
    topic_slug = get_topic_slug(week_id)

    missing = _missing_solution_files(repo, week_id)
    if missing:
        pytest.skip(
            f"dev/{topic_slug}/homework/reference_solution is missing autograder solution_files: "
            f"{', '.join(missing)}. Add these files to run reference-container validation."
        )

    run_sh = repo / topic_slug / "homework" / "container" / "run.sh"
    if not run_sh.is_file():
        pytest.skip(f"No container run.sh for week {week_id} at {run_sh}")

    env = {**os.environ, "DOCKER_BUILDKIT": "1", "REPO_ROOT": str(repo.resolve())}
    env.pop("GRADING_STUDENT_SUBMISSION", None)

    proc = subprocess.run(
        ["bash", str(run_sh)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=_CONTAINER_TIMEOUT_SEC,
        env=env,
    )
    if proc.returncode != 0:
        combined = f"{proc.stdout}\n{proc.stderr}"
        msg = (
            f"Homework container for week {week_id} ({topic_slug}) exited {proc.returncode}.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        if "NotImplementedError" in combined:
            msg += (
                "\n\nHint: NotImplementedError usually means reference_solution was not merged into "
                "solutions/ (e.g. GRADING_STUDENT_SUBMISSION=1 in the container), or stubs are still "
                "used because dev reference is incomplete vs autograder.yaml solution_files."
            )
        pytest.fail(msg)
