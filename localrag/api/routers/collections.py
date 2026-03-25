from __future__ import annotations

from fastapi import APIRouter, Depends

from localrag.api import service as api_service
from localrag.api.dependencies import get_collection_repository, get_ingestion_service
from localrag.api.repository import ChromaCollectionRepository
from localrag.api.schemas import (
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionNamePath,
    RebuildCollectionRequest,
    RebuildCollectionResponse,
)
from localrag.ingestion.service import IngestionService

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=CollectionListResponse)
def list_collections(
    collection_repo: ChromaCollectionRepository = Depends(get_collection_repository),
) -> CollectionListResponse:
    return api_service.list_collections_response(collection_repo)


@router.delete("/{name}", response_model=CollectionDeleteResponse)
def delete_collection(
    name: CollectionNamePath,
    collection_repo: ChromaCollectionRepository = Depends(get_collection_repository),
) -> CollectionDeleteResponse:
    return api_service.delete_collection_response(collection_repo, name)


@router.post("/rebuild", response_model=RebuildCollectionResponse)
def rebuild_collection(
    request: RebuildCollectionRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> RebuildCollectionResponse:
    return api_service.rebuild_collection_response(request, ingestion_service)
