"""Retrieval quality checks against the "Why We Sleep" book corpus.

These exercise the real ingest + retrieval path against a populated Chroma
collection, which the unit suite cannot cover: it fakes the vector store, so
neither embedding quality nor the Chroma call contract is checked there.

The collection is built by ingesting the EPUB:

    localrag --set chroma_collection_name=why-we-sleep ingest <book>.epub

Skipped unless that collection exists and is populated.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from localrag.application.container import get_retriever
from localrag.settings import load_settings, set_current_settings

pytestmark = pytest.mark.integration

COLLECTION = os.getenv("LOCALRAG_BOOK_COLLECTION", "why-we-sleep")
CHROMA_PATH = os.getenv("CHROMA_PERSIST_PATH", "/app/data/chroma")

# Topics the book covers at length, paired with terms that any correct
# retrieval for that topic should surface. Kept to substantive claims from
# the text rather than incidental wording, so a reasonable chunking or
# embedding change does not break them spuriously.
TOPIC_EXPECTATIONS: list[tuple[str, tuple[str, ...]]] = [
    ("What is adenosine and how does it create sleep pressure?", ("adenosine",)),
    ("How does caffeine affect sleep?", ("caffeine",)),
    ("What is the difference between REM and NREM sleep?", ("rem", "nrem")),
    ("How does melatonin regulate the circadian rhythm?", ("melatonin", "circadian")),
    ("What does alcohol do to REM sleep?", ("alcohol",)),
    ("How does sleep deprivation affect memory?", ("memory",)),
]


@pytest.fixture(scope="module")
def collection() -> Any:
    chromadb = pytest.importorskip("chromadb")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        handle = client.get_collection(COLLECTION)
    except Exception as exc:  # chroma raises bare errors for missing or unreadable stores
        pytest.skip(f"collection {COLLECTION!r} unavailable at {CHROMA_PATH}: {exc}")
    if handle.count() == 0:
        pytest.skip(f"collection {COLLECTION!r} is empty; ingest the EPUB first")
    return handle


@pytest.fixture(scope="module")
def retriever(collection: Any) -> Any:
    """LocalRAG's own retriever, so queries embed through the configured provider.

    Querying Chroma with raw ``query_texts`` would make it download and run its
    own default embedding model, which is not the model the corpus was built
    with — and therefore not what production retrieval does.
    """
    # Depends on `collection` so these tests skip, rather than error, when the
    # corpus is absent. Assert on it here so the dependency is not dead weight.
    assert collection.count() > 0
    set_current_settings(load_settings().with_overrides(chroma_collection_name=COLLECTION))
    return get_retriever()


def _joined_text(chunks: list[dict[str, Any]]) -> str:
    return " ".join(str(chunk.get("text", "")) for chunk in chunks).lower()


def test_collection_is_populated(collection: Any) -> None:
    """A full book must produce a substantial number of chunks, not a handful."""
    assert collection.count() > 100


def test_every_chunk_carries_source_metadata(collection: Any) -> None:
    """Sources must be attributable; a chunk without a source cannot be cited."""
    raw = collection.get(include=["metadatas"], limit=500)
    metadatas = raw.get("metadatas") or []
    assert metadatas, "expected metadata for the sampled chunks"
    missing = [meta for meta in metadatas if not (meta or {}).get("source")]
    assert not missing, f"{len(missing)} sampled chunks have no source metadata"


def test_chunks_come_from_the_epub(collection: Any) -> None:
    """The EPUB must be routed through a parser that records it as the source."""
    raw = collection.get(include=["metadatas"], limit=200)
    sources = {str((meta or {}).get("source", "")) for meta in raw.get("metadatas") or []}
    assert any(source.lower().endswith(".epub") for source in sources), sources


def test_parsed_text_is_prose_not_markup(collection: Any) -> None:
    """EPUB conversion must yield readable prose, not leftover XHTML."""
    raw = collection.get(include=["documents"], limit=200)
    documents = [doc for doc in (raw.get("documents") or []) if doc]
    assert documents, "expected documents in the collection"

    markup_bearing = [doc for doc in documents if "<html" in doc.lower() or "<div" in doc.lower()]
    assert not markup_bearing, f"{len(markup_bearing)}/{len(documents)} chunks still contain markup"

    # A book chunk that is mostly whitespace or a few characters signals a
    # chunking bug rather than a legitimately short heading.
    tiny = [doc for doc in documents if len(doc.strip()) < 20]
    assert len(tiny) < len(documents) * 0.1, f"{len(tiny)}/{len(documents)} chunks are near-empty"


def test_chunk_index_is_contiguous(collection: Any) -> None:
    """Chunk indices must cover 0..n-1 with no gaps, or ordering and citations drift."""
    raw = collection.get(include=["metadatas"], limit=collection.count())
    indices = sorted(
        int((meta or {}).get("chunk_index", -1)) for meta in raw.get("metadatas") or []
    )
    assert indices, "expected chunk_index metadata"
    assert indices == list(range(len(indices))), "chunk_index is not contiguous from 0"


@pytest.mark.parametrize(("question", "expected_terms"), TOPIC_EXPECTATIONS)
def test_semantic_retrieval_surfaces_expected_topic(
    retriever: Any,
    question: str,
    expected_terms: tuple[str, ...],
) -> None:
    """Retrieval must return chunks about the topic asked."""
    chunks = retriever.retrieve(question, n_results=5)
    assert chunks, f"no chunks retrieved for {question!r}"

    haystack = _joined_text(chunks)
    hits = [term for term in expected_terms if term in haystack]
    assert hits, f"none of {expected_terms} found in top-5 chunks for {question!r}"


def test_retrieved_chunks_are_attributable(retriever: Any) -> None:
    """Every retrieved chunk needs a source, or the answer cannot be cited."""
    chunks = retriever.retrieve("What is REM sleep?", n_results=3)
    assert chunks, "no chunks retrieved"
    assert all(str(chunk.get("source", "")) for chunk in chunks), "a chunk has no source"


def test_retrieval_respects_n_results(retriever: Any) -> None:
    """The caller's top-k must be honoured; overruns break context-window budgeting."""
    for requested in (1, 3, 8):
        chunks = retriever.retrieve("sleep and dreaming", n_results=requested)
        assert len(chunks) <= requested, f"asked for {requested}, got {len(chunks)}"


def test_unrelated_query_returns_different_chunks(retriever: Any) -> None:
    """A query far from the book's subject must not return the same chunks.

    Guards against a degenerate index that returns effectively the same result
    for everything, which would still pass the per-topic assertions above.
    """
    on_topic = retriever.retrieve("REM sleep and dreaming", n_results=3)
    off_topic = retriever.retrieve(
        "quarterly VAT filing thresholds for freight brokers",
        n_results=3,
    )
    assert "sleep" in _joined_text(on_topic), "on-topic query missed sleep content"
    assert _joined_text(on_topic) != _joined_text(off_topic), (
        "unrelated queries returned identical chunks"
    )
