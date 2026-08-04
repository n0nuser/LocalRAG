"""Provider-neutral embedding contract.

Providers own batching and return one finite, non-empty vector per input, in input
order. ``batch_size`` is a provider hint and ``None`` means provider default.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


class EmbeddingError(Exception):
    """Base error for embedding failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Provider configuration is invalid or an optional dependency is absent."""


class EmbeddingTransportError(EmbeddingError):
    """The provider could not be reached."""


class EmbeddingResponseError(EmbeddingError, ValueError):
    """The provider returned an invalid response."""


class EmbeddingIncompatibilityError(EmbeddingError):
    """The provider cannot safely operate on a collection's embedding space."""


def validate_vectors(
    vectors: Sequence[Sequence[float]], *, provider: str, model: str
) -> list[list[float]]:
    """Validate shape and numeric values at the provider boundary."""
    result = [list(row) for row in vectors]
    dimension = 0
    for index, row in enumerate(result):
        if not row:
            message = f"{provider}/{model} returned an empty embedding vector at {index}"
            raise EmbeddingResponseError(message)
        if index == 0:
            dimension = len(row)
        elif len(row) != dimension:
            message = f"{provider}/{model} returned inconsistent vector dimensions"
            raise EmbeddingResponseError(message)
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in row
        ):
            message = f"{provider}/{model} returned a non-finite vector"
            raise EmbeddingResponseError(message)
    return [[float(value) for value in row] for row in result]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Stable interface shared by ingestion and retrieval embeddings."""

    provider_name: str
    model: str
    dimension: int | None
    timeout_seconds: float

    def embed(self, text: str, *, model: str | None = None) -> list[float]: ...

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        model: str | None = None,
    ) -> list[list[float]]: ...

    def close(self) -> None: ...
