"""Embedding provider contracts and implementations."""

from localrag.embedding.base import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingIncompatibilityError,
    EmbeddingProvider,
    EmbeddingResponseError,
    EmbeddingTransportError,
)

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingIncompatibilityError",
    "EmbeddingProvider",
    "EmbeddingResponseError",
    "EmbeddingTransportError",
]
