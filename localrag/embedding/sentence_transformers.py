"""Optional local sentence-transformers embedding provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from localrag.embedding.base import EmbeddingConfigurationError, validate_vectors


class SentenceTransformersProvider:
    """Local provider; install with ``uv sync --extra embedding``."""

    provider_name = "sentence-transformers"
    timeout_seconds = 0.0

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
                SentenceTransformer,
            )
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "sentence-transformers is required; install with `uv sync --extra embedding`"
            ) from exc
        self.model = model
        self._model: Any = SentenceTransformer(model)
        self.dimension: int | None = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        if model is not None and model != self.model:
            raise EmbeddingConfigurationError(
                "sentence-transformers model cannot be overridden per call"
            )
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        model: str | None = None,
    ) -> list[list[float]]:
        if model is not None and model != self.model:
            raise EmbeddingConfigurationError(
                "sentence-transformers model cannot be overridden per call"
            )
        if not texts:
            return []
        if batch_size is not None and batch_size < 1:
            raise EmbeddingConfigurationError("batch_size must be positive")
        vectors = self._model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=False, show_progress_bar=False
        )
        return validate_vectors(vectors, provider=self.provider_name, model=self.model)

    def close(self) -> None:
        return None
