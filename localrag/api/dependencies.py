from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from localrag.api.repository import ChromaCollectionRepository
from localrag.ingestion.embedder import OllamaEmbedder
from localrag.ingestion.service import IngestionService
from localrag.rag.engine import RAGEngine
from localrag.rag.retriever import Retriever
from localrag.settings import Settings, get_settings
from localrag.storage.vector_store import VectorStore


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


def get_api_settings() -> Settings:
    return get_settings()


def get_collection_repository(
    store: VectorStore = Depends(get_vector_store),
) -> ChromaCollectionRepository:
    return ChromaCollectionRepository(_vector_store=store)
