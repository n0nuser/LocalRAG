from __future__ import annotations

from fastapi import status

from localrag.application.errors import ApplicationError, ApplicationErrorKind


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


class AgentApiError(HttpMappedError):
    """Raised when the agent endpoint cannot run (e.g. missing provider credentials)."""


def to_http_error(exc: ApplicationError) -> HttpMappedError:
    status_by_kind = {
        ApplicationErrorKind.BAD_REQUEST: status.HTTP_400_BAD_REQUEST,
        ApplicationErrorKind.FORBIDDEN: status.HTTP_403_FORBIDDEN,
        ApplicationErrorKind.NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ApplicationErrorKind.BAD_GATEWAY: status.HTTP_502_BAD_GATEWAY,
        ApplicationErrorKind.SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
        ApplicationErrorKind.TOO_MANY_REQUESTS: status.HTTP_429_TOO_MANY_REQUESTS,
        ApplicationErrorKind.UNSUPPORTED_MEDIA_TYPE: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        ApplicationErrorKind.INSUFFICIENT_STORAGE: status.HTTP_507_INSUFFICIENT_STORAGE,
        ApplicationErrorKind.REQUEST_ENTITY_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    }
    return HttpMappedError(status_by_kind[exc.kind], exc.detail)
