"""Load homework specs for oracle feedback."""

from pathlib import Path

from shared.week_config import get_repo_root, get_topic_slug


def load_homework_spec_text(week_id: str, max_chars: int = 8000) -> str:
    """Load homework spec from {topic}/homework/homework.ipynb via nbformat."""
    slug = get_topic_slug(week_id)
    repo_root = get_repo_root()
    nb_path = repo_root / slug / "homework" / "homework.ipynb"
    if not nb_path.exists():
        return ""

    import nbformat

    nb = nbformat.read(str(nb_path), as_version=4)
    parts: list[str] = []
    for cell in nb.cells:
        if cell.cell_type in ("markdown", "code"):
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            parts.append(src)
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...(truncated)"
    return text
