from __future__ import annotations

import re
from dataclasses import dataclass

from localrag.ingestion.loader import CODE_EXTENSIONS, MARKDOWN_EXTENSIONS
from localrag.settings import Settings

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    heading_path: str
    chunk_type: str


def chunk_document(text: str, file_type: str, settings: Settings) -> list[Chunk]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if file_type in MARKDOWN_EXTENSIONS:
        return _chunk_markdown(
            text=cleaned_text,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
    if file_type in CODE_EXTENSIONS:
        return _chunk_non_markdown(
            text=cleaned_text,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
            is_code=True,
        )
    return _chunk_non_markdown(
        text=cleaned_text,
        min_chars=settings.chunk_min_chars,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
        is_code=False,
    )


def _chunk_markdown(text: str, min_chars: int, max_chars: int, overlap_chars: int) -> list[Chunk]:
    lines = text.splitlines()
    heading_stack: list[str] = []
    sections: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_path = ""

    def flush_section() -> None:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((section_text, current_path))
        current_lines.clear()

    for line in lines:
        heading_match = _HEADING_PATTERN.match(line.strip())
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            current_path = " > ".join(heading_stack)
            current_lines.append(line)
            continue
        current_lines.append(line)
    flush_section()

    if not sections:
        return _chunk_non_markdown(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            is_code=False,
        )

    chunks: list[Chunk] = []
    for section_text, heading_path in sections:
        for item in _pack_blocks(
            section_text, min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars
        ):
            chunk_type = "markdown_section"
            if _looks_like_table(item):
                chunk_type = "markdown_table"
            elif _looks_like_fenced_code(item):
                chunk_type = "markdown_code"
            chunks.append(Chunk(text=item, heading_path=heading_path, chunk_type=chunk_type))
    return chunks


def _chunk_non_markdown(
    text: str, min_chars: int, max_chars: int, overlap_chars: int, *, is_code: bool
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for item in _pack_blocks(
        text, min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars, is_code=is_code
    ):
        chunk_type = "code_block" if is_code else "text_block"
        chunks.append(Chunk(text=item, heading_path="", chunk_type=chunk_type))
    return chunks


def _pack_blocks(  # noqa: C901
    text: str, min_chars: int, max_chars: int, overlap_chars: int = 0, *, is_code: bool = False
) -> list[str]:
    blocks = _split_blocks(text=text, is_code=is_code)
    if not blocks:
        return []

    effective_max_chars = max(1, max_chars)
    effective_min_chars = max(1, min(min_chars, effective_max_chars))

    packed: list[str] = []
    current = ""
    for block_text, block_kind in blocks:
        normalized = block_text.strip()
        if not normalized:
            continue

        if len(normalized) > effective_max_chars and block_kind == "paragraph":
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_long_paragraph(normalized, effective_max_chars, overlap_chars))
            continue

        candidate = normalized if not current else f"{current}\n\n{normalized}"
        if len(candidate) <= effective_max_chars:
            current = candidate
            continue

        if current:
            packed.append(current)
        current = normalized

    if current:
        packed.append(current)

    merged: list[str] = []
    for part in packed:
        if not merged:
            merged.append(part)
            continue
        previous = merged[-1]
        if (
            len(previous) < effective_min_chars
            and len(previous) + len(part) + 2 <= effective_max_chars
        ):
            merged[-1] = f"{previous}\n\n{part}"
            continue
        merged.append(part)
    return merged


def _split_blocks(text: str, *, is_code: bool) -> list[tuple[str, str]]:  # noqa: C901, PLR0912, PLR0915
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            fence_lines = [line]
            index += 1
            while index < len(lines):
                fence_line = lines[index]
                fence_lines.append(fence_line)
                index += 1
                if fence_line.strip().startswith("```"):
                    break
            blocks.append(("\n".join(fence_lines), "code"))
            continue

        if _is_table_row(stripped):
            table_lines = [line]
            index += 1
            while index < len(lines) and _is_table_row(lines[index].strip()):
                table_lines.append(lines[index])
                index += 1
            blocks.append(("\n".join(table_lines), "table"))
            continue

        if is_code and _is_top_level_symbol(stripped):
            code_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if _is_top_level_symbol(next_line.strip()):
                    break
                code_lines.append(next_line)
                index += 1
            blocks.append(("\n".join(code_lines), "paragraph"))
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            next_stripped = next_line.strip()
            if not next_stripped or next_stripped.startswith("```") or _is_table_row(next_stripped):
                break
            if is_code and _is_top_level_symbol(next_stripped):
                break
            paragraph_lines.append(next_line)
            index += 1
        blocks.append(("\n".join(paragraph_lines), "paragraph"))
    return blocks


def _split_long_paragraph(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text_len = len(text)
    safe_overlap = max(0, min(overlap_chars, max_chars - 1))
    chunks: list[str] = []
    start = 0
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            end = _snap_to_boundary(text, start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - safe_overlap if end < text_len else text_len
    return chunks


def _snap_to_boundary(text: str, start: int, end: int) -> int:
    window = text[start:end]
    sentence_breaks = [start + m.start() for m in _SENTENCE_BOUNDARY_PATTERN.finditer(window)]
    if sentence_breaks:
        return sentence_breaks[-1]
    last_space = window.rfind(" ")
    if last_space > 0:
        return start + last_space
    return end


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and "|" in line[1:]


def _is_top_level_symbol(line: str) -> bool:
    return line.startswith(("def ", "class ", "async def "))


def _looks_like_table(text: str) -> bool:
    lines = text.splitlines()
    return any(_is_table_row(line.strip()) for line in lines)


def _looks_like_fenced_code(text: str) -> bool:
    return "```" in text
