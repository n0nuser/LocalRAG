"""A metadata filter narrows the BM25 candidate set before top-k against real Chroma.

Chroma's ``where`` already restricts the dense path, so the result's sources alone
cannot show whether the lexical path honoured the filter. The settings give the
lexical list all of the relevance weight, so its order decides the ranking; an
empty lexical list leaves both private chunks tied at zero.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from localrag.application.container import get_ingestion_service, get_retriever
from localrag.application.runtime import clear_runtime_caches, get_vector_store
from localrag.settings import Settings, get_settings, load_settings, set_current_settings

pytestmark = pytest.mark.integration

# Each paragraph becomes its own chunk at this chunk size, so the public file
# holds many chunks that each outrank the private file's only relevant chunk.
PUBLIC_TEXT = "\n\n".join(["lantern lantern lantern lantern"] * 8)
PRIVATE_TEXT = "the lantern was lit\n\nnothing relevant here"


def _reset_runtime() -> None:
    get_retriever.cache_clear()
    get_ingestion_service.cache_clear()
    clear_runtime_caches()


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    previous = get_settings()
    configured = load_settings().with_overrides(
        chroma_persist_path=str(tmp_path / "chroma"),
        chroma_collection_name="hybrid-metadata-filter",
        embedding_cache_path=str(tmp_path / "embedding-cache"),
        ingest_roots=[str(tmp_path)],
        chunking_mode="recursive",
        chunk_max_chars=32,
        chunk_min_chars=1,
        chunk_overlap_chars=0,
        retrieval_mode="hybrid",
        bm25_weight=1.0,
        freshness_weight=0.0,
        rerank_enabled=False,
        parent_expansion_enabled=False,
    )
    set_current_settings(configured)
    _reset_runtime()
    yield configured
    _reset_runtime()
    set_current_settings(previous)


@pytest.mark.usefixtures("settings")
def test_hybrid_filter_keeps_lexical_hits_outranked_by_filtered_out_chunks(
    tmp_path: Path,
) -> None:
    public = tmp_path / "public.txt"
    public.write_text(PUBLIC_TEXT, encoding="utf-8")
    private = tmp_path / "private.txt"
    private.write_text(PRIVATE_TEXT, encoding="utf-8")
    private_source = str(private.resolve())
    for path in (public, private):
        get_ingestion_service().ingest_file(path)
    stored = get_vector_store().collection.get(include=["documents", "metadatas"])
    stored_by_source = sorted(
        (str(metadata["source"]), str(document))
        for document, metadata in zip(
            stored["documents"] or [], stored["metadatas"] or [], strict=True
        )
    )
    assert stored_by_source == sorted(
        [(str(public.resolve()), "lantern lantern lantern lantern")] * 8
        + [(private_source, "the lantern was lit"), (private_source, "nothing relevant here")]
    )

    contexts = get_retriever().retrieve(
        "lantern", n_results=2, metadata_filter={"source": private_source}
    )

    # A positive fused score means the chunk was ranked by the lexical list, the
    # only list with weight; the dense path alone would score both chunks zero.
    # The term-free chunk is in that list too, because BM25 returns zero-score rows
    # to fill top_k.
    assert [(context["source"], context["text"], context["score"] > 0) for context in contexts] == [
        (private_source, "the lantern was lit", True),
        (private_source, "nothing relevant here", True),
    ]
