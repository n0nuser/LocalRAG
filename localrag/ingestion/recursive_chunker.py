from __future__ import annotations

from localrag.ingestion.contract import Chunk

_SEPARATORS = ("\n\n", "\n", " ", "")


def chunk_document(text: str, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Split text recursively on paragraph, line, word, then character boundaries.

    A non-empty atomic token that exceeds ``max_chars`` is emitted intact and
    marked ``oversized`` rather than silently discarded. Empty input emits no
    chunks. Results retain source order; duplicate text remains distinct by index.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    limit = max(1, max_chars)
    pieces = _split(cleaned, limit)
    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if current and len(candidate) > limit:
            packed.append(current)
            overlap = max(0, min(overlap_chars, len(current) - 1))
            current = current[-overlap:] + " " + piece if overlap else piece
        else:
            current = candidate
    if current:
        packed.append(current)

    return [
        Chunk(
            text=piece,
            chunk_type="recursive",
            chunk_index=index,
            metadata={"oversized": len(piece) > limit},
        )
        for index, piece in enumerate(packed)
    ]


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    separator = next((item for item in _SEPARATORS if item and item in text), "")
    if not separator:
        return [text]
    parts = [part.strip() for part in text.split(separator) if part.strip()]
    if len(parts) == 1:
        return [text]
    result: list[str] = []
    for part in parts:
        result.extend(_split(part, limit))
    return result
