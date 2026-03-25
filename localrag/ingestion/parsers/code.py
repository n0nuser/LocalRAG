from __future__ import annotations

from pathlib import Path


def parse_code(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()
