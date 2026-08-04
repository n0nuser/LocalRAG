from __future__ import annotations

from localrag.ingestion.contract import stable_chunk_id
from localrag.ingestion.recursive_chunker import chunk_document


def test_recursive_chunking_is_ordered_and_preserves_duplicate_pieces() -> None:
    chunks = chunk_document("alpha beta alpha beta", max_chars=10, overlap_chars=0)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.text for chunk in chunks] == ["alpha beta", "alpha beta"]
    assert all(chunk.start_offset is None and chunk.end_offset is None for chunk in chunks)


def test_recursive_chunking_handles_empty_and_oversized_atomic_input() -> None:
    assert chunk_document(" \n ", max_chars=4, overlap_chars=0) == []

    chunks = chunk_document("abcdefghij", max_chars=4, overlap_chars=0)

    assert len(chunks) == 1
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].metadata["oversized"] is True


def test_chunk_id_is_stable_and_distinguishes_duplicate_positions() -> None:
    first = stable_chunk_id("doc.md", "recursive", 0, "same")
    repeated = stable_chunk_id("doc.md", "recursive", 0, "same")
    second = stable_chunk_id("doc.md", "recursive", 1, "same")

    assert first == repeated
    assert first != second
