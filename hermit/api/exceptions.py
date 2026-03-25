from __future__ import annotations


class IngestApiError(Exception):
    """Raised when HTTP ingest rules reject a path; map to an HTTP error in the API layer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
