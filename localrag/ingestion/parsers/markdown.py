from __future__ import annotations

from pathlib import Path


def parse_markdown(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    if content.startswith("---"):
        lines = content.splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            for index in range(1, len(lines)):
                if lines[index].strip() == "---":
                    return "".join(lines[index + 1 :]).strip()

    return content.strip()
