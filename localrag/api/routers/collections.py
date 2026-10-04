from __future__ import annotations

from fastapi import APIRouter, Depends

from localrag.api import service as api_service
from localrag.api.dependencies import get_collection_repository, get_container, require_api_key
from localrag.api.schemas import (
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionNamePath,
    RebuildCollectionRequest,
    RebuildCollectionResponse,
)
from localrag.application.container import Container
from localrag.application.repository import ChromaCollectionRepository

router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    dependencies=[Depends(require_api_key)],
)


@router.get("", response_model=CollectionListResponse)
def list_collections(
    collection_repo: ChromaCollectionRepository = Depends(get_collection_repository),
) -> CollectionListResponse:
    return api_service.list_collections_response(collection_repo)


@router.delete("/{name}", response_model=CollectionDeleteResponse)
def delete_collection(
    name: CollectionNamePath,
    container: Container = Depends(get_container),
) -> CollectionDeleteResponse:
    response = api_service.delete_collection_response(container.collection_repository, name)
    container.invalidate()
    return response


@router.post("/rebuild", response_model=RebuildCollectionResponse)
def rebuild_collection(
    request: RebuildCollectionRequest,
    container: Container = Depends(get_container),
) -> RebuildCollectionResponse:
    response = api_service.rebuild_collection_response(request, container.ingestion_service)
    container.invalidate()
    return response
