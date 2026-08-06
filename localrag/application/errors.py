from __future__ import annotations

from enum import StrEnum


class ApplicationErrorKind(StrEnum):
    BAD_REQUEST = "bad_request"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    BAD_GATEWAY = "bad_gateway"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TOO_MANY_REQUESTS = "too_many_requests"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    INSUFFICIENT_STORAGE = "insufficient_storage"
    REQUEST_ENTITY_TOO_LARGE = "request_entity_too_large"


class ApplicationError(Exception):
    """Transport-neutral failure raised by an application use case."""

    def __init__(self, kind: ApplicationErrorKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class IngestError(ApplicationError):
    """Raised when an ingestion use case rejects or cannot process input."""


class QueryError(ApplicationError):
    """Raised when a query use case cannot retrieve or generate an answer."""
