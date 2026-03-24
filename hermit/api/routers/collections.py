from __future__ import annotations

from fastapi import APIRouter, Depends

from hermit.api.dependencies import get_vector_store
from hermit.storage.vector_store import VectorStore

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("")
def list_collections(vector_store: VectorStore = Depends(get_vector_store)) -> dict[str, object]:
    return {"collections": vector_store.list_collections()}


@router.delete("/{name}")
def delete_collection(
    name: str, vector_store: VectorStore = Depends(get_vector_store)
) -> dict[str, str]:
    vector_store.delete_collection(name)
    return {"status": "ok"}
