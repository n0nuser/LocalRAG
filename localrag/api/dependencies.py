from __future__ import annotations

from functools import lru_cache
from http import HTTPStatus

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from localrag.api.repository import ChromaCollectionRepository
from localrag.ingestion.embedder import OllamaEmbedder
from localrag.ingestion.service import IngestionService
from localrag.rag.engine import RAGEngine
from localrag.rag.retriever import Retriever
from localrag.settings import Settings, get_settings
from localrag.storage.vector_store import VectorStore

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore.create(
        persist_path=settings.chroma_persist_path,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache(maxsize=1)
def get_embedder() -> OllamaEmbedder:
    settings = get_settings()
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(settings=settings, embedder=get_embedder(), vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    settings = get_settings()
    return RAGEngine(settings=settings, retriever=get_retriever())


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        settings=settings, embedder=get_embedder(), vector_store=get_vector_store()
    )


def require_api_key(
    key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Enforce X-API-Key when API_KEY is configured. No-op when API_KEY is empty."""
    configured = settings.api_key
    if not configured:
        return
    if not key or key != configured:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def get_api_settings() -> Settings:
    return get_settings()


def get_collection_repository(
    store: VectorStore = Depends(get_vector_store),
) -> ChromaCollectionRepository:
    return ChromaCollectionRepository(_vector_store=store)
