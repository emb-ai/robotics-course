"""Batch-safe Docker execution for reports-only grading."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .models import HomeworkSpec


@dataclass
class DockerRunResult:
    exit_code: int
    stdout: str
    stderr: str
    elapsed_sec: float
    pytest_xml: str


def prebuild_homework_image(spec: HomeworkSpec, repo_root: str | Path | None = None) -> None:
    """Build the homework test image once before parallel student runs."""

    root = Path(repo_root) if repo_root is not None else _repo_root_from_spec(spec)
    override = _write_override_yaml(spec, root / ".autograder-tmp" / "batch-build")
    cmd = [
        "docker",
        "compose",
        "-f",
        str(Path(spec.compose_file)),
        "-f",
        str(override),
        "build",
        "homework-tests",
    ]
    subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=True, env=_safe_env(root))


def run_student_tests(
    spec: HomeworkSpec,
    student_id: str,
    files: dict[str, str],
    selected_tests: list[str],
    result_dir: str | Path,
    repo_root: str | Path | None = None,
) -> DockerRunResult:
    """Run selected tests for one student and write pytest XML under ``result_dir``."""

    root = Path(repo_root) if repo_root is not None else _repo_root_from_spec(spec)
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    tmp_base = root / ".autograder-tmp" / "batch"
    tmp_base.mkdir(parents=True, exist_ok=True)
    timeout_sec = int(spec.limits.get("timeout_sec", 120))
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="student_", dir=tmp_base) as tmpdir:
        solutions_dir = Path(tmpdir).resolve()
        (solutions_dir / "__init__.py").write_text("", encoding="utf-8")
        for name, content in files.items():
            target = (solutions_dir / name).resolve()
            target.relative_to(solutions_dir)
            target.write_text(content, encoding="utf-8", errors="replace")

        override = _write_override_yaml(spec, solutions_dir)
        cmd = [
            "docker",
            "compose",
            "-p",
            _project_name(spec.id, student_id),
            "-f",
            str(Path(spec.compose_file)),
            "-f",
            str(override),
            "run",
            "--rm",
            "-e",
            "GRADING_STUDENT_SUBMISSION=1",
            "-e",
            f"GRADING_TEST_TIMEOUT_SEC={timeout_sec}",
            "-v",
            f"{solutions_dir}:{_solutions_mount_path(spec)}",
            "-v",
            f"{result_path.resolve()}:/results",
            "homework-tests",
            *selected_tests,
            "--junitxml=/results/pytest.xml",
        ]
        overhead = int(os.environ.get("AUTOGRADER_DOCKER_OVERHEAD_SEC", "600"))
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_sec + overhead,
            env=_safe_env(root),
        )

    xml_path = result_path / "pytest.xml"
    pytest_xml = xml_path.read_text(encoding="utf-8", errors="replace") if xml_path.exists() else ""
    return DockerRunResult(
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_sec=time.monotonic() - started,
        pytest_xml=pytest_xml,
    )


def _write_override_yaml(spec: HomeworkSpec, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "override.yaml"
    path.write_text(
        "\n".join(
            [
                "services:",
                "  homework-tests:",
                f"    mem_limit: {int(spec.limits.get('memory_mb', 512))}m",
                f"    cpus: \"{spec.limits.get('cpus', 1)}\"",
                f"    network_mode: {spec.limits.get('network', 'none')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _safe_env(repo_root: Path) -> dict[str, str]:
    allowed = ("PATH", "HOME", "USER", "TMPDIR", "DOCKER_HOST", "DOCKER_CONFIG", "XDG_RUNTIME_DIR")
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["DOCKER_BUILDKIT"] = "1"
    env["REPO_ROOT"] = str(repo_root.resolve())
    return env


def _solutions_mount_path(spec: HomeworkSpec) -> str:
    return f"/app/{spec.topic_slug}/homework/solutions"


def _repo_root_from_spec(spec: HomeworkSpec) -> Path:
    homework_dir = Path(spec.homework_dir).resolve()
    return homework_dir.parents[1]


def _project_name(homework_id: str, student_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", student_id.lower()).strip("-") or "student"
    return f"batch-{homework_id}-{slug}-{next(tempfile._get_candidate_names())}"[:63]
