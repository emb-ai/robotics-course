"""File tools using rtk for token-optimized output (60-90% reduction)."""

import subprocess
from pathlib import Path


def _rtk_available() -> bool:
    try:
        subprocess.run(["rtk", "--version"], capture_output=True, timeout=2)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_directory(path: str | Path, base: Path | None = None) -> str:
    """Return compact directory listing. Uses rtk ls when available."""
    path = Path(path)
    if base and not path.is_absolute():
        path = base / path
    if not path.exists():
        return f"Path not found: {path}"
    if not path.is_dir():
        return str(path)
    try:
        if _rtk_available():
            r = subprocess.run(
                ["rtk", "ls", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(base) if base else None,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout.strip()
        # Fallback: simple listing
        lines = []
        for p in sorted(path.iterdir())[:50]:
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{p.name}{suffix}")
        return "\n".join(lines) if lines else "(empty)"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str | Path, level: str = "normal", base: Path | None = None) -> str:
    """
    Read file with optional compression. Uses rtk read when available.
    level: "normal" (default) or "aggressive" (signatures only for code).
    """
    path = Path(path)
    if base and not path.is_absolute():
        path = base / path
    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"Not a file: {path}"
    try:
        if _rtk_available():
            args = ["rtk", "read", str(path)]
            if level == "aggressive":
                args.extend(["-l", "aggressive"])
            r = subprocess.run(args, capture_output=True, text=True, timeout=10, cwd=str(base) if base else None)
            if r.returncode == 0 and r.stdout:
                return r.stdout.strip()
        # Fallback: plain read
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error: {e}"


def search_code(query: str, path: str | Path = ".", base: Path | None = None, max_chars: int = 2000) -> str:
    """
    Search code. Uses rtk grep when available for grouped, compact results.
    """
    path = Path(path) if path != "." else Path(".")
    abs_path = (base / path) if base else path.resolve()
    if not abs_path.exists():
        return f"Path not found: {abs_path}"
    try:
        if _rtk_available():
            r = subprocess.run(
                ["rtk", "grep", query, str(abs_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode in (0, 1) and r.stdout:
                out = r.stdout.strip()
                return out[:max_chars] + ("..." if len(out) > max_chars else "")
        # Fallback: ripgrep
        r = subprocess.run(
            ["rg", "-i", "-C", "1", "-n", query, str(abs_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode not in (0, 1):
            return "Search failed."
        return (r.stdout or r.stderr or "")[:max_chars] or "No matches."
    except FileNotFoundError:
        return "rg not found. Install ripgrep or rtk."
    except Exception as e:
        return f"Error: {e}"
