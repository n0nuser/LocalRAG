from __future__ import annotations


class RetrievalError(Exception):
    """Raised when embedding or vector query cannot complete; mapped to HTTP in the API layer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
