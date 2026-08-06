from __future__ import annotations

from functools import lru_cache
from typing import cast

from localrag.application.jobs import JobRegistry
from localrag.application.repository import ChromaCollectionRepository
from localrag.application.runtime import (
    get_bm25_index,
    get_embedder,
    get_vector_store,
)
from localrag.embedding.cache import EmbeddingCache
from localrag.ingestion.service import IngestionService
from localrag.llm.factory import build_provider
from localrag.plugins.retriever import ManagedRetriever, discover_retriever_plugins
from localrag.rag.engine import RAGEngine
from localrag.rag.query_cache import QueryCache
from localrag.rag.retriever import Retriever
from localrag.settings import get_settings


@lru_cache(maxsize=1)
def get_retriever() -> ManagedRetriever:
    settings = get_settings()
    registry = discover_retriever_plugins()
    return cast(
        "ManagedRetriever",
        ManagedRetriever(registry, registry.create(settings.retriever_plugin, settings)),
    )


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    settings = get_settings()
    return RAGEngine(
        settings=settings,
        retriever=cast("Retriever", get_retriever()),
        provider=build_provider(settings),
    )


@lru_cache(maxsize=1)
def get_query_cache() -> QueryCache:
    settings = get_settings()
    return QueryCache(
        maxsize=settings.query_cache_maxsize,
        ttl_seconds=settings.query_cache_ttl_seconds,
    )


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        settings=settings,
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25_index(),
        embedding_cache=EmbeddingCache(
            settings.embedding_cache_path,
            max_entries=settings.embedding_cache_max_entries,
            max_bytes=settings.embedding_cache_max_bytes,
            preprocessing_version=settings.embedding_cache_preprocessing_version,
            task_prefix=settings.embedding_cache_task_prefix,
        ),
    )


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    return JobRegistry()


def get_collection_repository() -> ChromaCollectionRepository:
    return ChromaCollectionRepository(_vector_store=get_vector_store())
