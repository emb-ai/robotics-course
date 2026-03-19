"""Common env var helpers for tools config."""

import os


def env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v else default


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("true", "1", "yes")
