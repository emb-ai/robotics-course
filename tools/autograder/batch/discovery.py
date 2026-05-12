"""Discover homework configs for local reports-only batch grading."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shared.week_config import get_repo_root as default_repo_root

from .models import HomeworkSpec


_HOMEWORK_ID_RE = re.compile(r"^(\d+)")


@dataclass
class DiscoveryResult:
    homeworks: dict[str, HomeworkSpec] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def discover_homeworks(repo_root: str | Path | None = None) -> DiscoveryResult:
    """Discover every top-level ``*/homework/autograder.yaml`` under the repo."""

    root = Path(repo_root) if repo_root is not None else default_repo_root()
    result = DiscoveryResult()
    for yaml_path in sorted(root.glob("*/homework/autograder.yaml")):
        topic_slug = yaml_path.parent.parent.name
        homework_id = _homework_id(topic_slug)
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            result.warnings.append(f"{yaml_path}: failed to parse autograder.yaml: {exc}")
            continue

        if not isinstance(raw, dict):
            result.warnings.append(f"{yaml_path}: expected mapping, got {type(raw).__name__}")
            raw = {}

        spec = _spec_from_config(root, topic_slug, homework_id, raw, result.warnings)
        result.homeworks[homework_id] = spec
    return result


def get_homework(repo_root: str | Path | None, homework_id: str) -> HomeworkSpec:
    """Return one discovered homework or raise ``ValueError`` with useful context."""

    result = discover_homeworks(repo_root)
    key = str(homework_id)
    if key not in result.homeworks:
        available = ", ".join(sorted(result.homeworks)) or "none"
        raise ValueError(f"Unknown homework {homework_id!r}; available: {available}")
    return result.homeworks[key]


def _spec_from_config(
    root: Path,
    topic_slug: str,
    homework_id: str,
    raw: dict[str, Any],
    warnings: list[str],
) -> HomeworkSpec:
    homework_dir = root / topic_slug / "homework"
    compose_file = homework_dir / "container" / "docker_compose.yaml"
    if not compose_file.is_file():
        warnings.append(f"{topic_slug}: missing docker_compose.yaml at {compose_file}")

    solution_files = _string_list(raw.get("solution_files"), f"{topic_slug}: solution_files", warnings)
    problem_ids = _string_dict(raw.get("problem_ids"), f"{topic_slug}: problem_ids", warnings)
    points = {
        problem_id: float(value)
        for problem_id, value in (raw.get("points") or {}).items()
        if isinstance(problem_id, str) and isinstance(value, (int, float))
    }
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    limits_raw = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    limits = {
        "timeout_sec": limits_raw.get("timeout_sec", 120),
        "memory_mb": limits_raw.get("memory_mb", 512),
        "cpus": limits_raw.get("cpus", 1),
        "network": limits_raw.get("network", "none"),
    }
    test_dependencies = _test_dependencies(raw.get("test_dependencies"), problem_ids, warnings, topic_slug)

    return HomeworkSpec(
        id=homework_id,
        topic_slug=topic_slug,
        homework_dir=str(homework_dir),
        compose_file=str(compose_file),
        solution_files=solution_files,
        problem_ids=problem_ids,
        points=points,
        metrics=metrics,
        limits=limits,
        test_dependencies=test_dependencies,
    )


def _homework_id(topic_slug: str) -> str:
    match = _HOMEWORK_ID_RE.match(topic_slug)
    return match.group(1) if match else topic_slug


def _string_list(value: Any, label: str, warnings: list[str]) -> list[str]:
    if value is None:
        warnings.append(f"{label}: missing list")
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        warnings.append(f"{label}: expected list of strings")
        return []
    return list(value)


def _string_dict(value: Any, label: str, warnings: list[str]) -> dict[str, str]:
    if value is None:
        warnings.append(f"{label}: missing mapping")
        return {}
    if not isinstance(value, dict):
        warnings.append(f"{label}: expected mapping")
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            result[key] = item
        else:
            warnings.append(f"{label}: ignored non-string entry {key!r}: {item!r}")
    return result


def _test_dependencies(
    raw: Any,
    problem_ids: dict[str, str],
    warnings: list[str],
    topic_slug: str,
) -> dict[str, list[str]]:
    dependencies = {test_file: [f"{problem_id}.py"] for test_file, problem_id in problem_ids.items()}
    if raw is None:
        return dependencies
    if not isinstance(raw, dict):
        warnings.append(f"{topic_slug}: test_dependencies must be a mapping")
        return dependencies
    for test_file, values in raw.items():
        if not isinstance(test_file, str) or not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            warnings.append(f"{topic_slug}: ignored malformed test_dependencies entry {test_file!r}")
            continue
        dependencies[test_file] = list(values)
    return dependencies
