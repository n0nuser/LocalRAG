from __future__ import annotations

from pathlib import Path


def parse_anydoc(path: Path) -> str:
    """Convert an office/document file through the optional Rust-backed binding."""
    try:
        import anydoc  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "This file type requires the optional anydoc parser; install with "
            "`uv sync --extra anydoc`."
        ) from exc
    return str(anydoc.to_markdown(str(path)))
