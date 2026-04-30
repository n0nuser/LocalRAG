from __future__ import annotations

import logging
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from localrag.logging_config import request_id_ctx

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind ``X-Request-ID`` (or generate one) and log each request when it finishes."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        header_rid = request.headers.get("x-request-id")
        rid = header_rid or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            status_code = response.status_code if response is not None else 500
            _slog.info(
                "http_request",
                component="middleware",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(elapsed_ms, 1),
                correlation_id=rid,
            )
            request_id_ctx.reset(token)
