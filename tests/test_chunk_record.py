from __future__ import annotations

from typing import Any

import pytest

from localrag.chunks.record import retrieval_context


def test_retrieval_context_from_a_vector_hit_carries_the_stored_row_and_distance() -> None:
    stored = {"source": "/docs/a.md", "chunk_index": 1, "chunk_id": "id-1", "ingested_at": "t"}

    context = retrieval_context("text", stored, score=0.5, distance=1.0)

    assert context == {
        "text": "text",
        "chunk_id": "id-1",
        "source": "/docs/a.md",
        "chunk_index": 1,
        "score": 0.5,
        "distance": 1.0,
        "ingested_at": "t",
        "metadata": stored,
    }


def test_retrieval_context_prefers_the_vector_store_id_over_the_stored_one() -> None:
    context = retrieval_context("text", {"chunk_id": "stored-id"}, score=0.1, chunk_id="chroma-id")

    assert context["chunk_id"] == "chroma-id"


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param({}, id="empty-row"),
        pytest.param({"chunk_index": "first", "source": 7, "ingested_at": 0}, id="ill-typed-row"),
        pytest.param({"author": "someone", "page": 3}, id="foreign-row"),
    ],
)
def test_retrieval_context_reads_partial_or_foreign_rows_with_fallbacks(
    stored: dict[str, Any],
) -> None:
    context = retrieval_context("text", stored, score=0.25)

    assert context == {
        "text": "text",
        "source": "unknown",
        "chunk_index": -1,
        "score": 0.25,
        "ingested_at": None,
        "metadata": stored,
    }
