from __future__ import annotations

from fastapi import APIRouter, Depends

from localrag.api import service as api_service
from localrag.api.dependencies import get_api_settings, get_collection_repository
from localrag.api.repository import ChromaCollectionRepository
from localrag.api.schemas import HealthResponse
from localrag.settings import Settings

router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_api_settings),
    collection_repo: ChromaCollectionRepository = Depends(get_collection_repository),
) -> HealthResponse:
    return api_service.check_health(settings, collection_repo)
