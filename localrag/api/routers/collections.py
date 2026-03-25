from __future__ import annotations

from fastapi import APIRouter, Depends

from localrag.api import service as api_service
from localrag.api.dependencies import get_collection_repository
from localrag.api.repository import ChromaCollectionRepository
from localrag.api.schemas import (
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionNamePath,
)

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
