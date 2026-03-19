"""Assert reference code in dev/ passes open + hidden tests in the homework container.

The container copies reference_solution/*.py into solutions/ unless GRADING_STUDENT_SUBMISSION=1
(autograder). run.sh does not set the latter, so this matches `homework/container/run.sh`.
"""

import subprocess
from pathlib import Path

import pytest

from shared.week_config import get_repo_root, get_topic_slug, list_weeks

# Cold Docker build + full pytest can take several minutes.
_CONTAINER_TIMEOUT_SEC = int(__import__("os").environ.get("HOMEWORK_REF_CONTAINER_TIMEOUT", "1200"))


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


def _reference_solution_populated(repo: Path, topic_slug: str) -> bool:
    ref = repo / "dev" / topic_slug / "homework" / "reference_solution"
    if not ref.is_dir():
        return False
    try:
        return any(ref.iterdir())
    except OSError:
        return False


@pytest.fixture(scope="module")
def docker_ok():
    if not _docker_available():
        pytest.skip("Docker not available (docker info failed)")
    return True


@pytest.mark.integration
@pytest.mark.parametrize("week_id", list_weeks())
def test_reference_solution_passes_in_homework_container(week_id: str, docker_ok, repo_root):
    """Full grading run: open tests + hidden_tests + reference comparisons."""
    repo = Path(repo_root) if not isinstance(repo_root, Path) else repo_root
    topic_slug = get_topic_slug(week_id)
    if not _reference_solution_populated(repo, topic_slug):
        pytest.skip(
            f"dev/{topic_slug}/homework/reference_solution missing or empty; "
            "populate it to run reference-container validation."
        )
    run_sh = repo / topic_slug / "homework" / "container" / "run.sh"
    if not run_sh.is_file():
        pytest.skip(f"No container run.sh for week {week_id} at {run_sh}")

    proc = subprocess.run(
        ["bash", str(run_sh)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=_CONTAINER_TIMEOUT_SEC,
        env={
            **__import__("os").environ,
            "DOCKER_BUILDKIT": "1",
            "REPO_ROOT": str(repo.resolve()),
            # Do not set GRADING_STUDENT_SUBMISSION — container must use reference as solutions/.
        },
    )
    if proc.returncode != 0:
        msg = (
            f"Homework container for week {week_id} ({topic_slug}) exited {proc.returncode}.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        pytest.fail(msg)
