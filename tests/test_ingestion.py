from __future__ import annotations

from pathlib import Path

from hermit.ingestion.chunker import chunk_text
from hermit.ingestion.loader import list_supported_files


def test_chunk_text_returns_overlap_chunks() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_chars=10, overlap_chars=2)
    assert chunks
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ijklmnop")


def test_list_supported_files_non_recursive(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# title", encoding="utf-8")
    (docs / "nested").mkdir()
    (docs / "nested" / "b.md").write_text("nested", encoding="utf-8")

    files = list_supported_files(docs, recursive=False)

    assert [file.name for file in files] == ["a.md"]
