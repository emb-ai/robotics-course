"""Run grading in Docker with student code mounted."""

import os
import subprocess
import tempfile
from pathlib import Path

from .week_registry import (
    get_compose_path,
    get_limits,
    get_repo_root,
    get_solutions_mount_path,
    get_test_paths_for_submission,
)

# When the autograder runs inside a container with the host Docker socket,
# bind-mount paths passed to `docker compose run -v` must be host-absolute.
# Set HOST_REPO_ROOT to the repo path on the host so that tmp dirs are created
# under it and are therefore visible to the host daemon at the same path.
_HOST_REPO_ROOT: str | None = os.environ.get("HOST_REPO_ROOT")


def _write_override_yaml(week_id: str, tmpdir: Path) -> Path:
    """Write a docker-compose override with per-homework limits. Returns path to YAML."""
    limits = get_limits(week_id)
    network = limits.get("network", "none")
    yaml_path = tmpdir / "override.yaml"
    content = f"""# Auto-generated override from homework autograder.yaml
services:
  homework-tests:
    mem_limit: {limits.get('memory_mb', 512)}m
    cpus: "{limits.get('cpus', 1)}"
    network_mode: {network}
"""
    yaml_path.write_text(content)
    return yaml_path


def run(
    week_id: str,
    files: dict[str, str],
    timeout_sec: int | None = None,
    memory_mb: int | None = None,
    cpus: float | None = None,
) -> tuple[int, str, str]:
    """
    Run pytest in Docker with student files mounted over solutions/.
    Returns (exit_code, stdout, stderr).

    The effective time limit is subprocess.run(timeout=timeout_sec + 10): the host
    kills the docker compose process after that. The container itself has no internal
    timeout; limits.timeout_sec in autograder.yaml is passed as this subprocess timeout.
    """
    repo_root = get_repo_root()
    # host_root is the path used for Docker bind-mount arguments; equals repo_root
    # unless HOST_REPO_ROOT is set (containerized autograder with host Docker socket).
    host_root = Path(_HOST_REPO_ROOT) if _HOST_REPO_ROOT else repo_root
    compose_path = get_compose_path(week_id)
    mount_path = get_solutions_mount_path(week_id)
    limits = get_limits(week_id)
    timeout_sec = timeout_sec or limits.get("timeout_sec", 120)

    # Temp dirs live under host_root/.autograder-tmp/ so they share the same
    # absolute path on the host Docker daemon when running containerised.
    tmp_base = host_root / ".autograder-tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="autograder_", dir=tmp_base) as tmpdir:
        tmp = Path(tmpdir)
        # solutions/ is a Python package; ensure __init__.py for imports (flat weeks)
        (tmp / "__init__.py").write_text("", encoding="utf-8")
        for name, content in files.items():
            out_path = tmp / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8", errors="replace")

        override_path = _write_override_yaml(week_id, tmp)

        # Compose YAML lives under repo_root (/app in container). Map to host_root for the daemon.
        # Temp dir is already under host_root/.autograder-tmp/ — not a subpath of /app; use as-is.
        if _HOST_REPO_ROOT:
            host_compose_path = host_root / compose_path.relative_to(repo_root)
            host_override_path = override_path
            host_tmp = tmp
        else:
            host_compose_path = compose_path
            host_override_path = override_path
            host_tmp = tmp

        # Only run tests for submitted solution files so we don't import missing solution modules
        test_paths = get_test_paths_for_submission(week_id, files)
        cmd = [
            "docker",
            "compose",
            "-f",
            str(host_compose_path),
            "-f",
            str(host_override_path),
            "run",
            "--rm",
            "-e",
            "GRADING_STUDENT_SUBMISSION=1",
            "-v",
            f"{host_tmp}:{mount_path}",
            "homework-tests",
            *test_paths,
        ]

        # Outer timeout covers docker pull/build + container pytest; inner homework
        # limits.timeout_sec is mainly for documentation; subprocess must allow cold builds.
        overhead = int(os.environ.get("AUTOGRADER_DOCKER_OVERHEAD_SEC", "600"))
        repo_for_compose = str(host_root.resolve())
        proc = subprocess.run(
            cmd,
            cwd=str(host_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec + overhead,
            env={
                **dict(os.environ),
                "DOCKER_BUILDKIT": "1",
                "REPO_ROOT": repo_for_compose,
            },
        )

    return proc.returncode, proc.stdout, proc.stderr
