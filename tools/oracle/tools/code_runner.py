"""Run Python code in a restricted subprocess (timeout, temp dir)."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_python(
    code: str,
    timeout_sec: int = 10,
    stdin: str = "",
) -> tuple[int, str, str]:
    """
    Execute Python code in an isolated temp dir.
    Returns (exit_code, stdout, stderr). Output truncated to 10_000 chars each.
    """
    max_out = 10_000
    with tempfile.TemporaryDirectory(prefix="oracle_run_") as tmpdir:
        path = Path(tmpdir) / "main.py"
        path.write_text(code, encoding="utf-8", errors="replace")
        env = {**os.environ, "PYTHONPATH": tmpdir}
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=tmpdir,
                env=env,
            )
            out = (proc.stdout or "")[:max_out]
            err = (proc.stderr or "")[:max_out]
            return proc.returncode, out, err
        except subprocess.TimeoutExpired:
            return -1, "", f"Execution timed out after {timeout_sec}s"
