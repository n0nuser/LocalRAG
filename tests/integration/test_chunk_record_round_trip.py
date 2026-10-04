"""Ingest-written chunk metadata survives real Chroma and satisfies the retrieval read path.

The unit suite fakes the vector store, so only this test sees what Chroma
actually stores and returns: dropped keys, ``bool`` read back as ``int``, or
``int`` as ``float`` would all pass there and fail here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localrag.application.container import Container
from localrag.chunks.record import ChunkMetadata, SourceProvenance
from localrag.chunks.strategies import chunk_source
from localrag.settings import Settings, get_settings, load_settings, set_current_settings

pytestmark = pytest.mark.integration

# Recursive chunking over text with one unsplittable token yields both an
# oversized=False and an oversized=True chunk, so the boolean makes the trip both ways.
TEXT = "alpha beta gamma delta abcdefghijklmnopqrstuvwxyz epsilon zeta eta"


def _typed(row: dict[str, Any]) -> dict[str, tuple[type, Any]]:
    return {key: (type(value), value) for key, value in row.items()}


def _typed_without_mtime(row: dict[str, Any]) -> dict[str, tuple[type, Any]]:
    # Chroma round-trips float metadata to about 16 significant digits, so a
    # nanosecond-precision mtime comes back off in its last digit; it is
    # compared separately, within a microsecond.
    return {key: typed for key, typed in _typed(row).items() if key != "source_mtime"}


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    previous = get_settings()
    configured = load_settings().with_overrides(
        chroma_persist_path=str(tmp_path / "chroma"),
        chroma_collection_name="chunk-record-round-trip",
        embedding_cache_path=str(tmp_path / "embedding-cache"),
        ingest_roots=[str(tmp_path)],
        chunking_mode="recursive",
        chunk_max_chars=24,
        chunk_min_chars=1,
        chunk_overlap_chars=0,
    )
    set_current_settings(configured)
    yield configured
    set_current_settings(previous)


@pytest.fixture
def container(settings: Settings) -> Iterator[Container]:
    with Container.build(settings) as built:
        yield built


def test_ingested_metadata_round_trips_through_chroma_into_retrieval_contexts(
    tmp_path: Path, settings: Settings, container: Container
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text(TEXT, encoding="utf-8")
    source = str(path.resolve())

    container.ingestion_service.ingest_file(path)

    stored = container.vector_store.collection.get(include=["metadatas"])
    rows = {
        chunk_id: dict(metadata)
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"] or [], strict=True)
    }
    chunks = chunk_source(TEXT, ".txt", source, settings)
    assert {chunk.oversized for chunk in chunks} == {False, True}
    assert set(rows) == {chunk.chunk_id for chunk in chunks}
    for chunk in chunks:
        row = rows[chunk.chunk_id]
        provenance = SourceProvenance(
            file_type=".txt",
            ingested_at=str(row["ingested_at"]),
            content_hash=hashlib.sha256(TEXT.encode("utf-8")).hexdigest(),
            source_mtime=path.stat().st_mtime,
            git_commit="",
            tenant_id=settings.tenant_id,
        )
        expected = ChunkMetadata.of(chunk, provenance).to_stored()
        assert _typed_without_mtime(row) == _typed_without_mtime(expected)
        assert type(row["source_mtime"]) is float
        assert row["source_mtime"] == pytest.approx(expected["source_mtime"], abs=1e-6)

    contexts = container.retriever.retrieve("alpha beta gamma", n_results=5)

    assert contexts
    for context in contexts:
        row = rows[context["chunk_id"]]
        assert _typed(context["metadata"]) == _typed(row)
        assert (context["source"], context["chunk_index"]) == (source, row["chunk_index"])
