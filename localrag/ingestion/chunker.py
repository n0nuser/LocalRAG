from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        logger.debug("chunk_text_empty_input")
        return []
    if chunk_chars <= 0:
        logger.debug("chunk_text_single_chunk text_chars=%s", len(cleaned_text))
        return [cleaned_text]

    step = max(1, chunk_chars - max(0, overlap_chars))
    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        chunk = cleaned_text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    logger.debug(
        "chunk_text_done text_chars=%s chunk_chars=%s overlap=%s chunk_count=%s",
        len(cleaned_text),
        chunk_chars,
        overlap_chars,
        len(chunks),
    )
    return chunks
