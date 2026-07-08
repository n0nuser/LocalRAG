from __future__ import annotations

from dataclasses import dataclass

from localrag.rag.reranker import CrossEncoderReranker


@dataclass
class FakeModel:
    scores: list[float]

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == len(self.scores)
        return self.scores


def test_cross_encoder_reranker_reorders_by_score_and_trims_to_top_k() -> None:
    contexts = [
        {"text": "low relevance", "source": "a.md", "chunk_index": 0},
        {"text": "high relevance", "source": "b.md", "chunk_index": 0},
        {"text": "mid relevance", "source": "c.md", "chunk_index": 0},
    ]
    reranker = CrossEncoderReranker(model_name="unused", _model=FakeModel(scores=[0.1, 0.9, 0.5]))

    out = reranker.rerank("question", contexts, top_k=2)

    assert [c["source"] for c in out] == ["b.md", "c.md"]
    assert out[0]["rerank_score"] == 0.9


def test_cross_encoder_reranker_returns_empty_list_unchanged() -> None:
    reranker = CrossEncoderReranker(model_name="unused", _model=FakeModel(scores=[]))
    assert reranker.rerank("q", [], top_k=5) == []
