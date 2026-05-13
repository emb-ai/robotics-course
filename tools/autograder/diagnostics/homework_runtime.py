"""Runtime helpers for homework-specific diagnostics.

The batch diagnostics run inside the tools process, while student submissions
expect the homework directory to be importable as the current working context.
These helpers keep that setup scoped and remove homework modules afterwards so
different students/problems do not share imported `solutions` or `lib` modules.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import importlib.util
import multiprocessing as mp
import queue
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


HOMEWORK_MODULE_ROOTS = {"lib", "solutions", "reference_solution", "hidden_tests"}


@dataclass
class TimedCallResult:
    value: Any = None
    error: str | None = None
    traceback: str | None = None
    timeout: bool = False
    elapsed_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timeout


def homework_dir(context: Any) -> Path:
    return Path(context.repo_root) / str(context.topic_slug) / "homework"


def dev_homework_dir(context: Any) -> Path:
    return Path(context.repo_root) / "dev" / str(context.topic_slug) / "homework"


def reference_file(context: Any, filename: str) -> Path | None:
    candidates = [
        dev_homework_dir(context) / "reference_solution" / filename,
        homework_dir(context) / "reference_solution" / filename,
    ]
    return next((path for path in candidates if path.is_file()), None)


def hidden_dir(context: Any) -> Path | None:
    candidates = [
        dev_homework_dir(context) / "hidden_tests",
        homework_dir(context) / "hidden_tests",
    ]
    return next((path for path in candidates if path.exists()), None)


def submitted_file(context: Any, filename: str) -> Path | None:
    path = Path(context.normalized_submission_dir) / filename
    return path if path.is_file() else None


def import_paths(context: Any) -> list[Path]:
    submission_dir = Path(context.normalized_submission_dir)
    paths = [
        submission_dir,
        submission_dir.parent,
        homework_dir(context),
        dev_homework_dir(context),
    ]
    return [path for path in paths if path.exists()]


@contextlib.contextmanager
def isolated_homework_imports(context: Any, extra_paths: list[Path] | None = None) -> Iterator[None]:
    """Temporarily expose homework import roots and remove newly-loaded homework modules."""

    old_path = list(sys.path)
    before_modules = set(sys.modules)
    old_submission_dir = Path(context.normalized_submission_dir)
    with tempfile.TemporaryDirectory(prefix="diagnostic_homework_") as tmpdir:
        context.normalized_submission_dir = _prepare_runtime_homework(context, old_submission_dir, Path(tmpdir))
        paths = [Path(path) for path in (extra_paths or [])] + import_paths(context)
        for path in reversed(paths):
            raw = str(path)
            if raw not in sys.path:
                sys.path.insert(0, raw)
        try:
            yield
        finally:
            for name in list(sys.modules):
                if name in before_modules:
                    continue
                root = name.split(".", 1)[0]
                if root in HOMEWORK_MODULE_ROOTS or name.startswith("_diagnostic_"):
                    sys.modules.pop(name, None)
            context.normalized_submission_dir = old_submission_dir
            sys.path[:] = old_path


def _prepare_runtime_homework(context: Any, source_submission_dir: Path, runtime_root: Path) -> Path:
    """Create a temporary homework-shaped tree matching the Docker student mount."""

    real_hw = homework_dir(context)
    runtime_hw = runtime_root / str(context.topic_slug) / "homework"
    runtime_solutions = runtime_hw / "solutions"
    runtime_solutions.mkdir(parents=True, exist_ok=True)
    (runtime_solutions / "__init__.py").write_text("", encoding="utf-8")

    for filename in sorted(set(getattr(context, "submitted_files", []) or [])):
        source = source_submission_dir / filename
        if not source.is_file():
            continue
        destination = (runtime_solutions / filename).resolve()
        destination.relative_to(runtime_solutions.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for sibling in ("assets", "lib", "tests", "literature"):
        _mirror_homework_sibling(real_hw, runtime_hw, sibling)
    return runtime_solutions


def _mirror_homework_sibling(real_hw: Path, runtime_hw: Path, name: str) -> None:
    source = real_hw / name
    if not source.exists():
        return
    destination = runtime_hw / name
    if destination.exists():
        return
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def load_module_from_path(module_name: str, path: str | Path) -> Any:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_package_module_from_path(package_name: str, stem: str, path: Path) -> Any:
    package_root = str(path.parent.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    importlib.invalidate_caches()
    return importlib.import_module(f"{package_name}.{stem}")


def load_submitted_module(context: Any, filename: str, module_suffix: str = "") -> Any:
    path = submitted_file(context, filename)
    if path is None:
        raise FileNotFoundError(f"submitted file not found: {filename}")
    stem = Path(filename).stem
    if not module_suffix:
        return _import_package_module_from_path("solutions", stem, path)
    return load_module_from_path(f"_diagnostic_student_{context.student_id}_{stem}_{module_suffix}", path)


def load_reference_module(context: Any, filename: str, module_suffix: str = "") -> Any:
    path = reference_file(context, filename)
    if path is None:
        raise FileNotFoundError(f"reference file not found: {filename}")
    stem = Path(filename).stem
    if not module_suffix:
        return _import_package_module_from_path("reference_solution", stem, path)
    return load_module_from_path(f"_diagnostic_reference_{context.student_id}_{stem}_{module_suffix}", path)


def call_with_timeout(
    func: Callable[..., Any],
    *args: Any,
    timeout_sec: float = 0.0,
    **kwargs: Any,
) -> TimedCallResult:
    """Run a callable with a hard timeout and return a fail-soft result."""

    started = time.monotonic()
    if timeout_sec <= 0:
        try:
            return TimedCallResult(value=func(*args, **kwargs), elapsed_sec=time.monotonic() - started)
        except BaseException as exc:
            return TimedCallResult(
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
                elapsed_sec=time.monotonic() - started,
            )

    methods = mp.get_all_start_methods()
    method = "fork" if "fork" in methods else methods[0]
    ctx = mp.get_context(method)
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_call_worker, args=(result_queue, func, args, kwargs))
    proc.start()
    proc.join(timeout_sec)
    elapsed = time.monotonic() - started
    if proc.is_alive():
        proc.terminate()
        proc.join(0.5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return TimedCallResult(
            error=f"timeout after {timeout_sec:g}s",
            timeout=True,
            elapsed_sec=elapsed,
        )
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        return TimedCallResult(error="process exited without a result", elapsed_sec=elapsed)
    if status == "ok":
        return TimedCallResult(value=payload, elapsed_sec=elapsed)
    return TimedCallResult(
        error=payload.get("error", "unknown error"),
        traceback=payload.get("traceback"),
        elapsed_sec=elapsed,
    )


def _call_worker(result_queue: Any, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        result_queue.put(("ok", func(*args, **kwargs)))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        )


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
