from __future__ import annotations

from enum import StrEnum


class RetrievalFailureKind(StrEnum):
    BAD_GATEWAY = "bad_gateway"
    SERVICE_UNAVAILABLE = "service_unavailable"


class RetrievalError(Exception):
    """Raised when embedding or vector query cannot complete; mapped to HTTP in the API layer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.kind = (
            RetrievalFailureKind.SERVICE_UNAVAILABLE
            if status_code == 503
            else RetrievalFailureKind.BAD_GATEWAY
        )
