from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from localrag.api import service as api_service
from localrag.api.dependencies import get_api_settings, get_collection_repository
from localrag.api.repository import ChromaCollectionRepository
from localrag.api.schemas import HealthResponse, ReadinessResponse
from localrag.settings import Settings

router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def ready(
    response: Response,
    settings: Settings = Depends(get_api_settings),
    collection_repo: ChromaCollectionRepository = Depends(get_collection_repository),
) -> ReadinessResponse:
    result = api_service.check_readiness(settings, collection_repo)
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
