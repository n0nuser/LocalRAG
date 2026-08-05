from __future__ import annotations

from pathlib import Path

import anydoc


def parse_anydoc(path: Path) -> str:
    """Convert an office/document file through the required Rust-backed binding."""
    return str(anydoc.to_markdown(str(path)))


def detect_anydoc_format(path: Path) -> str | None:
    """Detect a supported document format from its content and path."""
    detected = anydoc.format_from_path(str(path))
    return str(detected) if detected is not None else None
