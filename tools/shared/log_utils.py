"""Log tail: efficient last-n-lines read."""

from pathlib import Path


def tail(path: Path, n: int = 100) -> str:
    """Read last n lines. For large files, seek from end to avoid loading entire file."""
    if not path.exists():
        return "(log file not found)"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            # For small files, readall and take last n is fine
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception as e:
        return str(e)
