from __future__ import annotations

from pathlib import Path

from docx import Document


def parse_docx(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs).strip()
