from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends

from hermit.api.dependencies import get_api_settings, get_vector_store
from hermit.config import Settings
from hermit.storage.vector_store import VectorStore

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
def health(
    settings: Settings = Depends(get_api_settings),
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict[str, object]:
    ollama_ok = False
    with httpx.Client(timeout=5.0) as client:
        try:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            ollama_ok = True
        except httpx.HTTPError:
            ollama_ok = False

    return {
        "status": "ok",
        "ollama_ok": ollama_ok,
        "chroma_path": settings.chroma_persist_path,
        "collections": vector_store.list_collections(),
    }
