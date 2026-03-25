from __future__ import annotations

import logging
from pathlib import Path

from hermit.ingestion.parsers.code import parse_code
from hermit.ingestion.parsers.docx import parse_docx
from hermit.ingestion.parsers.markdown import parse_markdown
from hermit.ingestion.parsers.pdf import parse_pdf
from hermit.ingestion.parsers.text import parse_text

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
TEXT_EXTENSIONS = {".txt", ".rst"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".css",
    ".html",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}
SUPPORTED_EXTENSIONS = (
    MARKDOWN_EXTENSIONS
    | TEXT_EXTENSIONS
    | CODE_EXTENSIONS
    | {
        ".pdf",
        ".docx",
    }
)

logger = logging.getLogger(__name__)


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_supported_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if is_supported_file(path) else []

    if recursive:
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    else:
        files = [candidate for candidate in path.glob("*") if candidate.is_file()]
    return [candidate for candidate in files if is_supported_file(candidate)]


def parse_file(path: Path) -> str:
    extension = path.suffix.lower()
    logger.debug("parse_file_dispatch path=%s extension=%s", path, extension)
    if extension == ".pdf":
        return parse_pdf(path)
    if extension == ".docx":
        return parse_docx(path)
    if extension in MARKDOWN_EXTENSIONS:
        return parse_markdown(path)
    if extension in CODE_EXTENSIONS:
        return parse_code(path)
    return parse_text(path)
