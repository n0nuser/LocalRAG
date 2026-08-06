from __future__ import annotations

from pathlib import Path

import pytest

from localrag.ingestion.chunker import chunk_text
from localrag.ingestion.loader import (
    UnsupportedFileTypeError,
    list_supported_files,
    parse_file,
)


def test_chunk_text_returns_overlap_chunks() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_chars=10, overlap_chars=2)
    assert chunks
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ijklmnop")


def test_chunk_text_with_non_positive_chunk_chars_returns_single_chunk() -> None:
    chunks = chunk_text("  abc  ", chunk_chars=0, overlap_chars=10)
    assert chunks == ["abc"]


def test_list_supported_files_non_recursive(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# title", encoding="utf-8")
    (docs / "nested").mkdir()
    (docs / "nested" / "b.md").write_text("nested", encoding="utf-8")

    files = list_supported_files(docs, recursive=False)

    assert [file.name for file in files] == ["a.md"]


def test_parse_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    """An unknown extension must fail loudly instead of being read as text."""
    path = tmp_path / "archive.xyz"
    path.write_text("not a document", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        parse_file(path)


def test_parse_file_rejects_binary_content(tmp_path: Path) -> None:
    """Binary content must be rejected even when the extension looks textual.

    Reading a binary file as text produces mojibake chunks that are embedded and
    become retrievable, poisoning the corpus without ever raising.
    """
    path = tmp_path / "payload.txt"
    path.write_bytes(b"PK\x03\x04\x00\x00text-like\x00\x00\xff\xfe binary")

    with pytest.raises(UnsupportedFileTypeError):
        parse_file(path)


def test_parse_file_still_reads_plain_text(tmp_path: Path) -> None:
    """The binary guard must not reject legitimate text files."""
    path = tmp_path / "notes.txt"
    path.write_text("plain prose with punctuation — and a dash", encoding="utf-8")

    assert "plain prose" in parse_file(path)
