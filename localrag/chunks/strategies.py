from __future__ import annotations

from localrag.chunks.fixed import chunk_text
from localrag.chunks.record import Chunk, ChunkDraft
from localrag.chunks.recursive import chunk_document as recursive_chunk_document
from localrag.chunks.structural import chunk_document as structural_chunk_document
from localrag.settings import Settings


def chunk_source(text: str, file_type: str, source: str, settings: Settings) -> list[Chunk]:
    """Split one source's text with the configured chunking strategy and place each chunk."""
    strategy = settings.chunking_mode
    if strategy == "fixed":
        drafts = [
            ChunkDraft(text=piece, chunk_type="fixed")
            for piece in chunk_text(
                text=text,
                chunk_chars=settings.chunk_chars,
                overlap_chars=settings.chunk_overlap_chars,
            )
        ]
    elif strategy == "recursive":
        drafts = recursive_chunk_document(
            text=text,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
    else:
        drafts = structural_chunk_document(text=text, file_type=file_type, settings=settings)
    return [
        Chunk.place(draft, source=source, chunk_index=index, strategy=strategy)
        for index, draft in enumerate(drafts)
    ]
