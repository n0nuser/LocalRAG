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
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        handle = client.get_collection(COLLECTION)
    except Exception:  # chroma raises bare errors for a missing collection name
        pytest.skip(f"collection {COLLECTION!r} is not present at {CHROMA_PATH}")
    if handle.count() == 0:
        pytest.skip(f"collection {COLLECTION!r} is empty; ingest the EPUB first")
    return handle


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


@pytest.mark.parametrize(("question", "expected_terms"), TOPIC_EXPECTATIONS)
def test_semantic_retrieval_surfaces_expected_topic(
    collection: Any,
    question: str,
    expected_terms: tuple[str, ...],
) -> None:
    """Embedding-based retrieval must return chunks about the topic asked."""
    result = collection.query(query_texts=[question], n_results=5)
    documents = (result.get("documents") or [[]])[0]
    assert documents, f"no chunks retrieved for {question!r}"

    haystack = " ".join(documents).lower()
    hits = [term for term in expected_terms if term in haystack]
    assert hits, f"none of {expected_terms} found in top-5 chunks for {question!r}"


def test_retrieval_is_ranked_by_distance(collection: Any) -> None:
    """Results must come back best-first; unordered distances break top-k cutoffs."""
    result = collection.query(query_texts=["Why do humans need to sleep?"], n_results=10)
    distances = (result.get("distances") or [[]])[0]
    assert len(distances) >= 2, "expected multiple results to compare"
    assert distances == sorted(distances), f"distances not ascending: {distances}"


def test_unrelated_query_is_less_similar_than_on_topic_query(collection: Any) -> None:
    """A query far from the book's subject must rank worse than an on-topic one.

    Guards against a degenerate index that returns near-identical distances for
    everything, which would still pass the per-topic assertions above.
    """
    on_topic = collection.query(query_texts=["REM sleep and dreaming"], n_results=1)
    off_topic = collection.query(
        query_texts=["quarterly VAT filing thresholds for freight brokers"],
        n_results=1,
    )
    best_on = (on_topic.get("distances") or [[]])[0][0]
    best_off = (off_topic.get("distances") or [[]])[0][0]
    assert best_on < best_off, f"on-topic {best_on} not closer than off-topic {best_off}"
