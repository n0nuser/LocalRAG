from __future__ import annotations

from functools import lru_cache

from localrag.embedding.base import EmbeddingProvider
from localrag.embedding.factory import build_embedding_provider
from localrag.rag.bm25_index import Bm25Index
from localrag.rag.reranker import CrossEncoderReranker
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
def get_embedder() -> EmbeddingProvider:
    return build_embedding_provider(get_settings())


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker | None:
    settings = get_settings()
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(model_name=settings.rerank_model)


@lru_cache(maxsize=1)
def get_bm25_index() -> Bm25Index:
    return Bm25Index.from_vector_store(get_vector_store())


def clear_runtime_caches() -> None:
    get_vector_store.cache_clear()
    get_embedder.cache_clear()
    get_reranker.cache_clear()
    get_bm25_index.cache_clear()


def settings_for_runtime() -> Settings:
    return get_settings()
