from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from localrag.rag.bm25_index import Bm25Hit
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


def _hit(source: str, score: float, age_days: float | None) -> dict[str, object]:
    hit: dict[str, object] = {
        "text": source,
        "source": source,
        "chunk_index": 0,
        "score": score,
        "metadata": {},
    }
    if age_days is not None:
        hit["ingested_at"] = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    return hit


def _hybrid_retriever(**overrides: object) -> Retriever:
    return Retriever(
        settings=Settings(retrieval_mode="hybrid", **overrides),  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
    )


def _fused_sources(retriever: Retriever, hits: list[dict[str, object]]) -> list[str]:
    fused = retriever._fuse_results(  # noqa: SLF001
        vector_hits=hits,
        bm25_hits=hits,
        top_k=len(hits),
    )
    return [str(hit["source"]) for hit in fused]


def test_hybrid_keeps_clearly_better_match_ahead_of_fresher_ones() -> None:
    """Recency must not overturn a clearly stronger relevance match.

    RRF scores are compressed — adjacent ranks differ by under 2% — so the
    previous multiplicative decay (spanning ~4600x over a year) ranked the best
    match last once it aged.
    """
    retriever = _hybrid_retriever()
    hits = [
        _hit("best_but_old", 0.99, 90),
        _hit("second", 0.98, 0),
        _hit("third", 0.90, 0),
        _hit("fourth", 0.89, 0),
    ]

    assert _fused_sources(retriever, hits)[0] == "best_but_old"


def test_hybrid_breaks_relevance_ties_by_recency() -> None:
    """Equally relevant candidates are ordered newest-first (ADR 006's intent)."""
    retriever = _hybrid_retriever()
    hits = [_hit("stale_policy", 0.9, 400), _hit("current_policy", 0.9, 1)]

    assert _fused_sources(retriever, hits)[0] == "current_policy"


def test_hybrid_does_not_penalise_candidates_without_ingested_at() -> None:
    """Missing metadata must neither reward nor punish a document."""
    retriever = _hybrid_retriever()
    hits = [_hit("undated_but_best", 0.99, None), _hit("dated_weaker", 0.98, 0)]

    assert _fused_sources(retriever, hits)[0] == "undated_but_best"


@pytest.mark.parametrize("overrides", [{"freshness_half_life_days": 0}, {"freshness_weight": 0.0}])
def test_hybrid_recency_can_be_disabled(overrides: dict[str, object]) -> None:
    retriever = _hybrid_retriever(**overrides)
    hits = [_hit("best_but_old", 0.99, 400), _hit("fresh_weaker", 0.98, 0)]

    assert _fused_sources(retriever, hits)[0] == "best_but_old"


def test_hybrid_reports_freshness_factor_without_rescoring() -> None:
    """freshness_factor stays observable even though it no longer moves the score."""
    retriever = _hybrid_retriever()
    contexts = [
        {
            "source": "a.md",
            "chunk_index": 0,
            "score": 0.5,
            "ingested_at": _hit("a", 0, 30)["ingested_at"],
        }
    ]

    out = retriever.apply_freshness(contexts, rescore=False)

    assert out[0]["score"] == 0.5
    assert out[0]["freshness_factor"] == pytest.approx(0.5, abs=0.01)


def test_hybrid_retrieve_does_not_rescore_by_freshness_end_to_end() -> None:
    """The full retrieve() path must not re-apply decay on top of fused scores.

    Guards the regression directly: fusing recency in and *then* multiplying by
    the decay factor is what buried the best match in the first place.
    """
    stale = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    fresh = datetime.now(UTC).isoformat()

    class Store:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k, where)
            return {
                "documents": [["best but old", "weaker but fresh"]],
                "metadatas": [
                    [
                        {"source": "best.md", "chunk_index": 0, "ingested_at": stale},
                        {"source": "weak.md", "chunk_index": 1, "ingested_at": fresh},
                    ]
                ],
                "distances": [[0.01, 0.5]],
            }

    class Bm25:
        @staticmethod
        def query(text: str, top_k: int) -> list[Bm25Hit]:
            _ = (text, top_k)
            return [
                Bm25Hit(
                    chunk_id="best.md:0",
                    text="best but old",
                    score=9.9,
                    metadata={"source": "best.md", "chunk_index": 0, "ingested_at": stale},
                ),
                Bm25Hit(
                    chunk_id="weak.md:1",
                    text="weaker but fresh",
                    score=1.0,
                    metadata={"source": "weak.md", "chunk_index": 1, "ingested_at": fresh},
                ),
            ]

    class Embedder:
        @staticmethod
        def embed_text(text: str, *, model: str | None = None) -> list[float]:
            _ = (text, model)
            return [0.1, 0.2, 0.3]

    retriever = Retriever(
        settings=Settings(retrieval_mode="hybrid", freshness_half_life_days=30),
        embedder=Embedder(),  # type: ignore[arg-type]
        vector_store=Store(),  # type: ignore[arg-type]
        bm25_index=Bm25(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("policy", n_results=2)

    assert contexts[0]["source"] == "best.md"
    # Still reported for observability, just no longer applied to the score.
    assert contexts[0]["freshness_factor"] < 0.01
