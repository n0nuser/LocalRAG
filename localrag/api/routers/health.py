from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from localrag.api import service as api_service
from localrag.api.dependencies import get_container
from localrag.api.schemas import HealthResponse, ReadinessResponse
from localrag.application.container import Container

router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def ready(
    response: Response,
    container: Container = Depends(get_container),
) -> ReadinessResponse:
    result = api_service.check_readiness(container.settings, container.collection_repository)
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
