from __future__ import annotations


class HttpMappedError(Exception):
    """Raised when a use case needs a specific HTTP response (mapped in ``main``)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class IngestApiError(HttpMappedError):
    """Raised when HTTP ingest rules reject a path."""


class RagApiError(HttpMappedError):
    """Raised when RAG query cannot complete (embedding or vector store failure)."""
