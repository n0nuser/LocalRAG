"""Optional cross-encoder reranking for retrieved contexts (not loaded unless enabled)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CrossEncoderModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _load_default_model(model_name: str) -> CrossEncoderModel:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415 — optional dependency

    return CrossEncoder(model_name)


@dataclass
class CrossEncoderReranker:
    model_name: str
    _model: CrossEncoderModel | None = field(default=None, repr=False)

    def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            self._model = _load_default_model(self.model_name)
        return self._model

    def rerank(
        self, question: str, contexts: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not contexts:
            return contexts
        pairs = [(question, str(context.get("text", ""))) for context in contexts]
        scores = self._get_model().predict(pairs)
        rescored = [
            {**context, "rerank_score": float(score)}
            for context, score in zip(contexts, scores, strict=True)
        ]
        rescored.sort(key=lambda context: context["rerank_score"], reverse=True)
        logger.debug("rerank_done candidates=%s top_k=%s", len(contexts), top_k)
        return rescored[:top_k]
