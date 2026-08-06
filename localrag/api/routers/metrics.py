from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from localrag.api.dependencies import require_api_key

router = APIRouter(prefix="", tags=["metrics"], dependencies=[Depends(require_api_key)])


@router.get("/metrics", summary="Prometheus metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in text format for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/build-info", summary="Build identity")
def build_info() -> dict[str, str]:
    """Expose the image identity so a running stack can be compared with source."""
    return {"build_sha": os.environ.get("LOCALRAG_BUILD_SHA", "unknown")}
