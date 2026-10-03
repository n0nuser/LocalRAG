"""Pin the exact chunk rows ingestion hands to the vector store.

Collections already on disk were written in this format, so it must not drift:
every key, every value, and every value's Python type (``False`` is not ``0``,
``1`` is not ``1.0``). The chunk-ID recipe is restated here on purpose rather
than imported, so a change to it fails this test instead of silently orphaning
stored IDs.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from localrag.ingestion import service as service_module
from localrag.ingestion.service import IngestionService
from localrag.settings import Settings

INGESTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
SOURCE_MTIME = 1_700_000_000.5


class FrozenClock:
    @staticmethod
    def now(tz: object = None) -> datetime:
        _ = tz
        return INGESTED_AT


class ConstantEmbedder:
    def embed_texts(
        self, texts: list[str], batch_size: int, *, model: str | None = None
    ) -> list[list[float]]:
        _ = (batch_size, model)
        return [[1.0] for _ in texts]


@dataclass
class CapturingVectorStore:
    writes: list[tuple[list[str], list[dict[str, Any]]]] = field(default_factory=list)

    def replace_source(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        _ = (source, embeddings)
        self.writes.append((chunks, metadatas))

    add_chunks = replace_source


def _chunk_id(source: str, strategy: str, index: int, text: str) -> str:
    identity = "\0".join(("1", source, strategy, str(index), text))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _typed(row: dict[str, Any]) -> dict[str, tuple[type, Any]]:
    return {key: (type(value), value) for key, value in row.items()}


def _expected_row(
    *,
    source: str,
    file_type: str,
    strategy: str,
    index: int,
    text: str,
    heading_path: str,
    chunk_type: str,
    content_hash: str,
    oversized: bool | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {} if oversized is None else {"oversized": oversized}
    row |= {
        "chunking_strategy": strategy,
        "source": source,
        "file_type": file_type,
        "chunk_index": index,
        "chunk_id": _chunk_id(source, strategy, index, text),
        "heading_path": heading_path,
        "chunk_type": chunk_type,
        "ingested_at": INGESTED_AT.isoformat(),
        "content_hash": content_hash,
        "source_mtime": SOURCE_MTIME,
        "git_commit": "",
        "tenant_id": "household",
    }
    return row


@pytest.mark.parametrize(
    ("file_name", "text", "settings_overrides", "expected_chunks"),
    [
        pytest.param(
            "notes.txt",
            "alpha beta gamma delta epsilon",
            {"chunking_mode": "fixed", "chunk_chars": 12, "chunk_overlap_chars": 2},
            [
                ("alpha beta g", "", "fixed", None),
                ("gamma delta", "", "fixed", None),
                ("ta epsilon", "", "fixed", None),
            ],
            id="fixed",
        ),
        pytest.param(
            "guide.md",
            "# Intro\n\nHello world.\n\n## Usage\n\n```python\nprint(1)\n```",
            {"chunking_mode": "structural", "chunk_max_chars": 1200, "chunk_min_chars": 1},
            [
                ("# Intro\n\nHello world.", "Intro", "markdown_section", None),
                ("## Usage\n\n```python\nprint(1)\n```", "Intro > Usage", "markdown_code", None),
            ],
            id="structural",
        ),
        pytest.param(
            "long.txt",
            "alpha beta gamma abcdefghijklmnopqrstuvwxyz",
            {
                "chunking_mode": "recursive",
                "chunk_max_chars": 16,
                "chunk_min_chars": 1,
                "chunk_overlap_chars": 0,
            },
            [
                ("alpha beta gamma", "", "recursive", False),
                ("abcdefghijklmnopqrstuvwxyz", "", "recursive", True),
            ],
            id="recursive",
        ),
    ],
)
def test_ingestion_writes_the_established_stored_chunk_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    text: str,
    settings_overrides: dict[str, Any],
    expected_chunks: list[tuple[str, str, str, bool | None]],
) -> None:
    monkeypatch.setattr(service_module, "datetime", FrozenClock)
    path = tmp_path / file_name
    path.write_text(text, encoding="utf-8")
    os.utime(path, (SOURCE_MTIME, SOURCE_MTIME))
    source = str(path.resolve())
    settings = Settings(ingest_roots=[str(tmp_path)], tenant_id="household", **settings_overrides)
    store = CapturingVectorStore()
    service = IngestionService(
        settings=settings,
        embedder=ConstantEmbedder(),  # type: ignore[arg-type]  # structural stand-in for the embedder seam
        vector_store=store,  # type: ignore[arg-type]  # captures writes instead of persisting them
    )

    service.ingest_file(path)

    strategy = settings_overrides["chunking_mode"]
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    expected_rows = [
        _expected_row(
            source=source,
            file_type=path.suffix,
            strategy=strategy,
            index=index,
            text=chunk_text,
            heading_path=heading_path,
            chunk_type=chunk_type,
            content_hash=content_hash,
            oversized=oversized,
        )
        for index, (chunk_text, heading_path, chunk_type, oversized) in enumerate(expected_chunks)
    ]
    [(documents, metadatas)] = store.writes
    assert documents == [chunk_text for chunk_text, *_ in expected_chunks]
    assert [_typed(row) for row in metadatas] == [_typed(row) for row in expected_rows]
