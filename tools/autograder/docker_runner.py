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
    use_gpu = os.environ.get("GRADING_USE_GPU") == "1"
    memory_mb = limits.get("memory_mb", 512)
    if use_gpu:
        memory_mb = max(int(memory_mb), 8192)
    gpu_block = "\n    gpus: all" if use_gpu else ""
    yaml_path = tmpdir / "override.yaml"
    content = f"""# Auto-generated override from homework autograder.yaml
services:
  homework-tests:
    mem_limit: {memory_mb}m
    cpus: "{limits.get('cpus', 1)}"
    network_mode: {network}{gpu_block}
"""
    yaml_path.write_text(content)
    return yaml_path


def _run_docker_compose(
    *,
    week_id: str,
    files: dict[str, str],
    host_tmp: Path,
    override_path: Path,
    host_root: Path,
    repo_root: Path,
    compose_path: Path,
    mount_path: str,
    timeout_sec: int,
) -> tuple[int, str, str]:
    if _HOST_REPO_ROOT:
        host_compose_path = host_root / compose_path.relative_to(repo_root)
    else:
        host_compose_path = compose_path

    test_paths = get_test_paths_for_submission(week_id, files)
    cmd = [
        "docker",
        "compose",
        "-f",
        str(host_compose_path),
        "-f",
        str(override_path),
        "run",
        "--rm",
        "-e",
        "GRADING_STUDENT_SUBMISSION=1",
        "-e",
        f"GRADING_TEST_TIMEOUT_SEC={timeout_sec}",
        "-v",
        f"{host_tmp}:{mount_path}",
        "homework-tests",
        *test_paths,
    ]

    overhead = int(os.environ.get("AUTOGRADER_DOCKER_OVERHEAD_SEC", "600"))
    repo_for_compose = str(host_root.resolve())
    safe_env: dict[str, str] = {
        k: v for k, v in os.environ.items()
        if k in ("PATH", "HOME", "USER", "TMPDIR", "DOCKER_HOST", "DOCKER_CONFIG", "XDG_RUNTIME_DIR")
    }
    safe_env["DOCKER_BUILDKIT"] = "1"
    safe_env["REPO_ROOT"] = repo_for_compose

    proc = subprocess.run(
        cmd,
        cwd=str(host_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec + overhead,
        env=safe_env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run(
    week_id: str,
    files: dict[str, str],
    timeout_sec: int | None = None,
    memory_mb: int | None = None,
    cpus: float | None = None,
    prepared_mount_dir: Path | None = None,
) -> tuple[int, str, str]:
    """
    Run pytest in Docker with student files mounted over solutions/.
    Returns (exit_code, stdout, stderr).

    When ``prepared_mount_dir`` is set, bind-mount that directory as-is (for staff
    local grading with package subdirs and binary assets). ``files`` is still used
    for partial test selection only.

    Test runtime is capped inside the container via ``timeout(1)`` and
    ``GRADING_TEST_TIMEOUT_SEC`` (from ``limits.timeout_sec`` in autograder.yaml).

    The host ``subprocess.run`` timeout is ``timeout_sec + AUTOGRADER_DOCKER_OVERHEAD_SEC``
    to allow first-time image build/pull; increase the env var if builds exceed it.
    """
    repo_root = get_repo_root()
    host_root = Path(_HOST_REPO_ROOT) if _HOST_REPO_ROOT else repo_root
    compose_path = get_compose_path(week_id)
    mount_path = get_solutions_mount_path(week_id)
    limits = get_limits(week_id)
    timeout_sec = timeout_sec or limits.get("timeout_sec", 120)

    tmp_base = host_root / ".autograder-tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)

    if prepared_mount_dir is not None:
        host_tmp = Path(prepared_mount_dir).resolve()
        override_path = _write_override_yaml(week_id, host_tmp.parent)
        return _run_docker_compose(
            week_id=week_id,
            files=files,
            host_tmp=host_tmp,
            override_path=override_path,
            host_root=host_root,
            repo_root=repo_root,
            compose_path=compose_path,
            mount_path=mount_path,
            timeout_sec=timeout_sec,
        )

    with tempfile.TemporaryDirectory(prefix="autograder_", dir=tmp_base) as tmpdir:
        tmp = Path(tmpdir)
        tmp_resolved = tmp.resolve()
        (tmp / "__init__.py").write_text("", encoding="utf-8")
        for name, content in files.items():
            out_path = (tmp / name).resolve()
            if not str(out_path).startswith(str(tmp_resolved) + os.sep) and out_path != tmp_resolved:
                import logging as _log
                _log.getLogger(__name__).warning("Skipping unsafe filename: %r", name)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8", errors="replace")

        override_path = _write_override_yaml(week_id, tmp)
        return _run_docker_compose(
            week_id=week_id,
            files=files,
            host_tmp=tmp,
            override_path=override_path,
            host_root=host_root,
            repo_root=repo_root,
            compose_path=compose_path,
            mount_path=mount_path,
            timeout_sec=timeout_sec,
        )
