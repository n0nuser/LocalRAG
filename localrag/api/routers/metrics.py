from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(prefix="", tags=["metrics"])


@router.get("/metrics", summary="Prometheus metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in text format for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
