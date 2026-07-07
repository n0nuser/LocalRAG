from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from localrag.rag.retriever import Retriever
from localrag.settings import Settings


@pytest.mark.parametrize(
    ("half_life_days", "age_days", "expected_factor"),
    [
        (0.0, 30, 1.0),
        (30.0, 30, 0.5),
    ],
)
def test_freshness_factor_matches_expected_decay(
    half_life_days: float, age_days: int, expected_factor: float
) -> None:
    retriever = Retriever(
        settings=Settings(freshness_half_life_days=half_life_days),
        embedder=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
    )
    ingested_at = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    contexts = [{"source": "doc.md", "chunk_index": 0, "score": 1.0, "ingested_at": ingested_at}]

    rescored = retriever.apply_freshness(contexts)

    assert rescored[0]["freshness_factor"] == pytest.approx(expected_factor, rel=0.05)


def test_freshness_ignores_invalid_timestamps() -> None:
    retriever = Retriever(
        settings=Settings(freshness_half_life_days=30.0),
        embedder=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
    )
    contexts = [
        {"source": "doc.md", "chunk_index": 0, "score": 1.0, "ingested_at": "bad-timestamp"}
    ]

    rescored = retriever.apply_freshness(contexts)

    assert rescored[0]["freshness_factor"] == 1.0


def test_freshness_decay_prefers_recent_chunk() -> None:
    stale = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    fresh = (datetime.now(UTC) - timedelta(days=2)).isoformat()

    class FreshnessStore:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k, where)
            return {
                "documents": [["old policy", "new policy"]],
                "metadatas": [
                    [
                        {"source": "policy.md", "chunk_index": 0, "ingested_at": stale},
                        {"source": "policy.md", "chunk_index": 1, "ingested_at": fresh},
                    ]
                ],
                "distances": [[0.01, 0.2]],
            }

    class StubEmbedder:
        @staticmethod
        def embed_text(text: str, *, model: str | None = None) -> list[float]:
            _ = (text, model)
            return [0.1, 0.2, 0.3]

    retriever = Retriever(
        settings=Settings(retrieval_mode="vector", freshness_half_life_days=30),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=FreshnessStore(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("refund policy", n_results=2)

    assert contexts[0]["chunk_index"] == 1
    assert contexts[0]["freshness_factor"] > contexts[1]["freshness_factor"]
