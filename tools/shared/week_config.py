"""Week registry: weeks.yaml + per-homework autograder.yaml with caching."""

import functools
from pathlib import Path
from typing import Any

import yaml


def _weeks_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "weeks.yaml"


def get_repo_root() -> Path:
    root = __import__("os").environ.get("AI_ROBOTICS_REPO_ROOT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=1)
def load_weeks_yaml() -> dict[str, Any]:
    with open(_weeks_path()) as f:
        return yaml.safe_load(f)


def _load_autograder_yaml(topic_slug: str) -> dict[str, Any]:
    """Load {topic}/homework/autograder.yaml. Returns {} if missing."""
    repo = get_repo_root()
    path = repo / topic_slug / "homework" / "autograder.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_week_config(week_id: str) -> dict[str, Any]:
    weeks = load_weeks_yaml()
    key = week_id
    if key not in weeks:
        key = int(week_id) if week_id.isdigit() else week_id
    if key not in weeks:
        raise ValueError(f"Unknown week_id: {week_id}")
    base = dict(weeks[key])
    topic_slug = base["topic_slug"]
    # Merge with per-homework autograder.yaml
    hw_cfg = _load_autograder_yaml(topic_slug)
    base.update(hw_cfg)
    # Auto-derive compose_file
    if "compose_file" not in base:
        base["compose_file"] = f"{topic_slug}/homework/container/docker_compose.yaml"
    return base


def get_topic_slug(week_id: str) -> str:
    return get_week_config(week_id)["topic_slug"]


def get_solution_files(week_id: str) -> list[str]:
    return list(get_week_config(week_id).get("solution_files", []))


def get_problem_ids(week_id: str) -> dict[str, str]:
    return dict(get_week_config(week_id).get("problem_ids", {}))


def get_points(week_id: str) -> dict[str, int]:
    """Points per problem_id. Default 1 if not specified."""
    return dict(get_week_config(week_id).get("points", {}))


def get_metrics_config(week_id: str) -> dict[str, dict[str, str]]:
    """Metrics config: problem_id -> {name, direction}."""
    return dict(get_week_config(week_id).get("metrics", {}))


def get_limits(week_id: str) -> dict[str, Any]:
    """Resource limits from per-homework config."""
    cfg = get_week_config(week_id).get("limits", {})
    return {
        "timeout_sec": cfg.get("timeout_sec", 120),
        "memory_mb": cfg.get("memory_mb", 512),
        "cpus": cfg.get("cpus", 1),
        "network": cfg.get("network", "none"),
    }


def list_weeks() -> list[str]:
    return [str(k) for k in load_weeks_yaml().keys()]


def list_weeks_with_homework_container(repo: Path | None = None) -> list[str]:
    """Week IDs from ``weeks.yaml`` that have ``homework/container/run.sh`` and ``docker_compose.yaml``.

    Use ``repo`` to resolve container paths (defaults to :func:`get_repo_root`). Week metadata still
    comes from ``weeks.yaml`` via :func:`get_topic_slug`.
    """
    root = repo if repo is not None else get_repo_root()
    out: list[str] = []
    for week_id in list_weeks():
        slug = get_topic_slug(week_id)
        container_dir = root / slug / "homework" / "container"
        if (container_dir / "run.sh").is_file() and (container_dir / "docker_compose.yaml").is_file():
            out.append(week_id)
    return out
