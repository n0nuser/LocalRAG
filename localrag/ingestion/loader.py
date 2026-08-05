from __future__ import annotations

import logging
from pathlib import Path

from localrag.ingestion.parsers.anydoc import detect_anydoc_format, parse_anydoc
from localrag.ingestion.parsers.code import parse_code
from localrag.ingestion.parsers.markdown import parse_markdown
from localrag.ingestion.parsers.pdf import parse_pdf
from localrag.ingestion.parsers.text import parse_text

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
# Formats whose parser emits Markdown even though the source file is not Markdown,
# so chunking should follow heading structure. Kept separate from
# MARKDOWN_EXTENSIONS because that set also selects the *parser* in
# parse_document(); a .pdf must still go to parse_pdf.
MARKDOWN_PRODUCING_EXTENSIONS = {".pdf"}
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
ANYDOC_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docm",
    ".docx",
    ".epub",
    ".ods",
    ".odp",
    ".odt",
    ".pot",
    ".pps",
    ".ppsm",
    ".ppsx",
    ".ppt",
    ".pptm",
    ".pptx",
    ".rtf",
    ".xls",
    ".xlsb",
    ".xlsm",
    ".xlsx",
}
SUPPORTED_EXTENSIONS = (
    MARKDOWN_EXTENSIONS
    | TEXT_EXTENSIONS
    | CODE_EXTENSIONS
    | {
        ".pdf",
        ".docx",
    }
    | ANYDOC_EXTENSIONS
)

logger = logging.getLogger(__name__)


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS or detect_anydoc_format(path) is not None


def list_supported_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if is_supported_file(path) else []

    if recursive:
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    else:
        files = [candidate for candidate in path.glob("*") if candidate.is_file()]
    return [candidate for candidate in files if is_supported_file(candidate)]


def detect_file_type(path: Path) -> str | None:
    """Return the parser type, preserving the dedicated PDF processing path."""
    detected_format = detect_anydoc_format(path)
    if detected_format == "pdf" or path.suffix.lower() == ".pdf":
        return "pdf"
    return detected_format or path.suffix.lower().removeprefix(".") or None


def parse_file(path: Path) -> str:
    extension = path.suffix.lower()
    file_type = detect_file_type(path)
    logger.debug(
        "parse_file_dispatch path=%s extension=%s detected_format=%s",
        path,
        extension,
        file_type,
    )
    if file_type == "pdf":
        return parse_pdf(path)
    if file_type in {suffix.removeprefix(".") for suffix in ANYDOC_EXTENSIONS}:
        return parse_anydoc(path)
    if extension in MARKDOWN_EXTENSIONS:
        return parse_markdown(path)
    if extension in CODE_EXTENSIONS:
        return parse_code(path)
    return parse_text(path)
