"""Shared utilities for week/submission parsing, usable without telegram."""

import re

WEEK_PATTERN = re.compile(r"(?:/grade|week)\s*(\d+)", re.I)


def extract_week(caption: str | None) -> str | None:
    """Extract zero-padded week ID from a caption/command string."""
    if not caption:
        return None
    m = WEEK_PATTERN.search(caption)
    if m:
        w = m.group(1)
        return w.zfill(2) if len(w) == 1 else w
    return None
