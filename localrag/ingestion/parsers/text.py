from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes


def parse_text(path: Path) -> str:
    raw = path.read_bytes()
    detected = from_bytes(raw).best()
    if detected is None:
        return raw.decode("utf-8", errors="replace").strip()
    return str(detected).strip()
