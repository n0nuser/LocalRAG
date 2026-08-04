"""Embedding provider construction."""

from __future__ import annotations

from localrag.embedding.base import EmbeddingConfigurationError, EmbeddingProvider
from localrag.ingestion.embedder import OllamaEmbedder
from localrag.settings import Settings


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured provider; Ollama remains the default."""
    provider = settings.embedding_provider.strip().lower()
    model = settings.effective_embedding_model
    if provider == "ollama":
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=model,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    if provider in {"sentence-transformers", "sentence_transformers"}:
        from localrag.embedding.sentence_transformers import (  # noqa: PLC0415
            SentenceTransformersProvider,
        )

        return SentenceTransformersProvider(settings.sentence_transformers_model or model)
    message = f"Unsupported embedding provider: {provider}"
    raise EmbeddingConfigurationError(message)
