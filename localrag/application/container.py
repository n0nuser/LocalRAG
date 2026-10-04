"""The composition root: the one place that builds LocalRAG's runtime objects.

Each process entry point builds one ``Container`` from its ``Settings`` and every
adapter asks it for collaborators: the API in its lifespan (``app.state.container``),
MCP in its server lifespan, the CLI once per command. One process therefore holds one
vector store, one BM25 index and one retriever, so what the API ingests is what MCP
and plugin retrieval read. A separate process (the MCP HTTP server, a CLI ingest)
has its own container and its own snapshot; ADR 035 already allows only one writer
per persist path.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from types import TracebackType
from typing import Any, Self, cast

from localrag.application.jobs import JobRegistry
from localrag.application.repository import ChromaCollectionRepository
from localrag.embedding.base import EmbeddingProvider
from localrag.embedding.cache import EmbeddingCache
from localrag.embedding.factory import build_embedding_provider
from localrag.ingestion.service import IngestionService
from localrag.llm.factory import build_provider
from localrag.llm.providers.base import BaseLLMProvider
from localrag.plugins.retriever import (
    ManagedRetriever,
    RetrieverContract,
    discover_retriever_plugins,
)
from localrag.rag.bm25_index import Bm25Index
from localrag.rag.engine import RAGEngine
from localrag.rag.query_cache import QueryCache
from localrag.rag.reranker import CrossEncoderReranker
from localrag.rag.retriever import Retriever
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore

_MISSING = object()


class Container:
    """Build each runtime object once, on first use, and share it.

    Construction is lazy so MCP ``initialize``/``tools/list`` and CLI commands that
    never retrieve stay cheap. FastAPI runs sync routes in a thread pool, so first use
    can race; one re-entrant lock serialises construction (a build reaches into other
    builds, hence ``RLock``) while already-built objects are read without it.

    Objects passed to ``build`` belong to the caller: the container shares them but
    never rebuilds or closes them.
    """

    def __init__(self, settings: Settings, provided: dict[str, Any]) -> None:
        self.settings = settings
        self._objects = provided
        self._provided = frozenset(provided)
        self._lock = threading.RLock()

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        engine: RAGEngine | None = None,
        query_cache: QueryCache | None = None,
        ingestion_service: IngestionService | None = None,
        job_registry: JobRegistry | None = None,
        collection_repository: ChromaCollectionRepository | None = None,
    ) -> Container:
        provided = {
            "vector_store": vector_store,
            "embedder": embedder,
            "engine": engine,
            "query_cache": query_cache,
            "ingestion_service": ingestion_service,
            "job_registry": job_registry,
            "collection_repository": collection_repository,
        }
        return cls(settings, {name: obj for name, obj in provided.items() if obj is not None})

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def vector_store(self) -> VectorStore:
        return self._shared(
            "vector_store",
            lambda: VectorStore.create(
                persist_path=self.settings.chroma_persist_path,
                collection_name=self.settings.chroma_collection_name,
            ),
        )

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._shared("embedder", lambda: build_embedding_provider(self.settings))

    @property
    def reranker(self) -> CrossEncoderReranker | None:
        return self._shared(
            "reranker",
            lambda: (
                CrossEncoderReranker(model_name=self.settings.rerank_model)
                if self.settings.rerank_enabled
                else None
            ),
        )

    @property
    def bm25_index(self) -> Bm25Index:
        return self._shared("bm25_index", lambda: Bm25Index.from_vector_store(self.vector_store))

    @property
    def retriever(self) -> ManagedRetriever:
        return self._shared("retriever", lambda: _managed_retriever(self))

    @property
    def llm_provider(self) -> BaseLLMProvider:
        return self._shared("llm_provider", lambda: build_provider(self.settings))

    @property
    def engine(self) -> RAGEngine:
        return self._shared(
            "engine", lambda: _engine(self.settings, self.retriever, self.llm_provider)
        )

    @property
    def query_cache(self) -> QueryCache:
        return self._shared(
            "query_cache",
            lambda: QueryCache(
                maxsize=self.settings.query_cache_maxsize,
                ttl_seconds=self.settings.query_cache_ttl_seconds,
            ),
        )

    @property
    def ingestion_service(self) -> IngestionService:
        return self._shared(
            "ingestion_service",
            lambda: IngestionService(
                settings=self.settings,
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                embedding_cache=EmbeddingCache(
                    self.settings.embedding_cache_path,
                    max_entries=self.settings.embedding_cache_max_entries,
                    max_bytes=self.settings.embedding_cache_max_bytes,
                    preprocessing_version=self.settings.embedding_cache_preprocessing_version,
                    task_prefix=self.settings.embedding_cache_task_prefix,
                ),
            ),
        )

    @property
    def job_registry(self) -> JobRegistry:
        return self._shared("job_registry", JobRegistry)

    @property
    def collection_repository(self) -> ChromaCollectionRepository:
        return self._shared(
            "collection_repository",
            lambda: ChromaCollectionRepository(_vector_store=self.vector_store),
        )

    def invalidate(self) -> None:
        """Make the next query see the collection as it is now, after any write.

        Ingest, rebuild and collection delete all end here. The retriever is closed
        and rebuilt so a plugin drops whatever it cached; the engine is rebuilt around
        it, keeping the LLM provider and its connection pool. The BM25 index is refreshed
        in place rather than replaced: the ingestion service holds the same instance,
        and replacing the ingestion service is not an option because its write lock
        is the only thing serialising writers inside one process (the persist-path
        file lock is re-entrant per process, not per thread).
        """
        with self._lock:
            retriever = self._discard("retriever")
            self._discard("engine")
            if retriever is not None:
                retriever.close()
            bm25_index = self._objects.get("bm25_index")
            if bm25_index is not None:
                bm25_index.refresh()
            query_cache = self._objects.get("query_cache")
            if query_cache is not None:
                query_cache.clear()

    def close(self) -> None:
        """Release what the container built; the caller still owns what it passed in."""
        with self._lock:
            retriever = self._discard("retriever")
            if retriever is not None:
                retriever.close()
            embedder = self._discard("embedder")
            if embedder is not None:
                embedder.close()
            self._objects = {name: self._objects[name] for name in self._provided}

    def _shared[T](self, name: str, build: Callable[[], T]) -> T:
        # One read outside the lock: a membership test followed by a second lookup
        # could lose the object to a concurrent ``invalidate`` in between.
        built = self._objects.get(name, _MISSING)
        if built is _MISSING:
            with self._lock:
                built = self._objects.get(name, _MISSING)
                if built is _MISSING:
                    built = build()
                    self._objects[name] = built
        return built

    def _discard(self, name: str) -> Any:
        if name in self._provided:
            return None
        return self._objects.pop(name, None)


def _builtin_retriever(container: Container, settings: Settings) -> RetrieverContract:
    return cast(
        "RetrieverContract",
        Retriever(
            settings=settings,
            embedder=container.embedder,
            vector_store=container.vector_store,
            bm25_index=container.bm25_index,
            reranker=container.reranker,
        ),
    )


def _managed_retriever(container: Container) -> ManagedRetriever:
    settings = container.settings
    registry = discover_retriever_plugins(builtin_factory=partial(_builtin_retriever, container))
    return cast(
        "ManagedRetriever",
        ManagedRetriever(registry, registry.create(settings.retriever_plugin, settings)),
    )


def _engine(
    settings: Settings, retriever: ManagedRetriever, provider: BaseLLMProvider
) -> RAGEngine:
    return RAGEngine(settings=settings, retriever=cast("Retriever", retriever), provider=provider)
