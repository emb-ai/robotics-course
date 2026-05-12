"""Scan one-folder-per-student local submissions."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from .models import SubmittedStudent


def scan_submissions(submissions_root: str | Path, allowed_solution_files: list[str]) -> list[SubmittedStudent]:
    """Scan top-level student directories for allowed Python solution files."""

    root = Path(submissions_root)
    allowed = set(allowed_solution_files)
    students: list[SubmittedStudent] = []
    for student_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        students.append(_scan_student(student_dir, allowed))
    return students


def _scan_student(student_dir: Path, allowed: set[str]) -> SubmittedStudent:
    files: dict[str, str] = {}
    ignored: list[str] = []
    errors: list[str] = []

    def add_file(name: str, content: str, origin: str) -> None:
        if name not in allowed:
            ignored.append(origin)
            return
        if name in files:
            errors.append(f"Duplicate solution filename {name!r} in {origin}")
            return
        files[name] = content

    for path in sorted(student_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(student_dir).as_posix()
        if path.suffix == ".py":
            add_file(path.name, path.read_text(encoding="utf-8", errors="replace"), rel)
        elif path.suffix == ".zip":
            _scan_zip(path, allowed, files, ignored, errors)
        else:
            ignored.append(rel)

    if errors:
        files = {}
    return SubmittedStudent(
        student_id=student_dir.name,
        source_dir=str(student_dir),
        files=files,
        ignored_files=ignored,
        errors=errors,
    )


def _scan_zip(
    zip_path: Path,
    allowed: set[str],
    files: dict[str, str],
    ignored: list[str],
    errors: list[str],
) -> None:
    try:
        with ZipFile(zip_path) as zf:
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                origin = f"{zip_path.name}:{info.filename}"
                if _unsafe_archive_name(info.filename):
                    errors.append(f"Unsafe archive entry {origin}")
                    continue
                name = PurePosixPath(info.filename).name
                if not name.endswith(".py"):
                    ignored.append(origin)
                    continue
                content = zf.read(info).decode("utf-8", errors="replace")
                if name not in allowed:
                    ignored.append(origin)
                    continue
                if name in files:
                    errors.append(f"Duplicate solution filename {name!r} in {origin}")
                    continue
                files[name] = content
    except BadZipFile as exc:
        errors.append(f"Invalid zip archive {zip_path.name}: {exc}")


def _unsafe_archive_name(name: str) -> bool:
    if not name or "\x00" in name:
        return True
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute():
        return True
    return any(part in {"", ".", ".."} for part in posix.parts)
