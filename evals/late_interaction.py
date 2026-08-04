"""Dependency-free late-interaction prototype for research issue #70.

This module deliberately accepts precomputed token embeddings. It is an
experiment seam, not an embedding provider or a Chroma-compatible retriever.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

Token = Sequence[float]
TokenMatrix = Sequence[Token]


def _validate_matrix(matrix: TokenMatrix, name: str) -> int:
    if not matrix:
        return 0
    dimension = len(matrix[0])
    if dimension == 0:
        message = f"{name} tokens must not be empty"
        raise ValueError(message)
    if any(len(token) != dimension for token in matrix):
        message = f"{name} tokens must have the same dimension"
        raise ValueError(message)
    return dimension


def maxsim(
    query: TokenMatrix,
    document: TokenMatrix,
    *,
    query_mask: Sequence[bool] | None = None,
    document_mask: Sequence[bool] | None = None,
    normalize: bool = False,
) -> float:
    """Score a query/document pair using ColBERT's sum-of-maxima rule."""
    query_dimension = _validate_matrix(query, "query")
    document_dimension = _validate_matrix(document, "document")
    if query_dimension and document_dimension and query_dimension != document_dimension:
        raise ValueError("query and document tokens must have the same dimension")
    if query_mask is not None and len(query_mask) != len(query):
        raise ValueError("query mask length must match query tokens")
    if document_mask is not None and len(document_mask) != len(document):
        raise ValueError("document mask length must match document tokens")

    query_tokens = [
        token for index, token in enumerate(query) if query_mask is None or query_mask[index]
    ]
    document_tokens = [
        token
        for index, token in enumerate(document)
        if document_mask is None or document_mask[index]
    ]
    if not query_tokens or not document_tokens:
        return 0.0
    score = sum(
        max(
            sum(left * right for left, right in zip(query_token, document_token, strict=True))
            for document_token in document_tokens
        )
        for query_token in query_tokens
    )
    return score / len(query_tokens) if normalize else score


class LateInteractionIndex:
    """Small in-memory token index with explicit JSON persistence."""

    schema_version = 1

    def __init__(self) -> None:
        self._documents: dict[str, tuple[tuple[float, ...], ...]] = {}

    def add(self, document_id: str, tokens: TokenMatrix) -> None:
        if document_id in self._documents:
            message = f"document id already exists: {document_id}"
            raise ValueError(message)
        _validate_matrix(tokens, "document")
        self._documents[document_id] = tuple(
            tuple(float(value) for value in token) for token in tokens
        )

    def search(
        self, query: TokenMatrix, *, top_k: int = 5, normalize: bool = False
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = (
            (document_id, maxsim(query, tokens, normalize=normalize))
            for document_id, tokens in self._documents.items()
        )
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "documents": dict(sorted(self._documents.items())),
        }
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> LateInteractionIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.schema_version or not isinstance(
            payload.get("documents"), dict
        ):
            raise ValueError("unsupported late-interaction index")
        index = cls()
        for document_id, tokens in payload["documents"].items():
            if not isinstance(document_id, str) or not isinstance(tokens, list):
                raise TypeError("malformed late-interaction document")
            index.add(document_id, tokens)
        return index
