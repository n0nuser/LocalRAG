"""One process, one set of runtime objects: what the API writes, MCP retrieval reads."""

from __future__ import annotations

import ast
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient
from fastmcp import Client

from localrag.api.dependencies import get_container
from localrag.api.main import app
from localrag.application.container import Container
from localrag.mcp.server import build_mcp_server
from localrag.rag.query_cache import QueryCache
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore

QUESTION = {"question": "quagga"}


@dataclass
class FakeEmbedder:
    provider_name: str = "fake"
    model: str = "fake-embed"
    dimension: int | None = None
    timeout_seconds: float = 1.0
    model_revision: str = ""
    closed: bool = False

    @staticmethod
    def embed(text: str, *, model: str | None = None) -> list[float]:
        _ = (text, model)
        return [1.0, 0.0]

    @staticmethod
    def embed_batch(
        texts: Sequence[str], *, batch_size: int | None = None, model: str | None = None
    ) -> list[list[float]]:
        _ = (batch_size, model)
        return [[1.0, 0.0] for _ in texts]

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeVectorStore:
    """One collection whose dense search finds nothing, so only BM25 can surface a chunk."""

    rows: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)

    def replace_source(
        self,
        source: str,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        _ = embeddings
        self.delete_by_source(source)
        for chunk_id, text, metadata in zip(ids, chunks, metadatas, strict=True):
            self.rows[chunk_id] = (text, metadata)

    def delete_by_source(self, source: str) -> None:
        for chunk_id in [key for key, (_, meta) in self.rows.items() if meta["source"] == source]:
            del self.rows[chunk_id]

    def list_distinct_sources(self) -> list[str]:
        return sorted({meta["source"] for _, meta in self.rows.values()})

    def get_all_chunks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [(chunk_id, text, meta) for chunk_id, (text, meta) in self.rows.items()]

    @staticmethod
    def list_collections() -> list[str]:
        return ["localrag"]

    def delete_collection(self, name: str) -> None:
        _ = name
        self.rows.clear()

    @staticmethod
    def query(
        embedding: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        _ = (embedding, top_k, where)
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        ingest_roots=[str(tmp_path)],
        chroma_persist_path=str(tmp_path / "chroma"),
        embedding_cache_path=str(tmp_path / "embedding-cache"),
        retrieval_mode="hybrid",
        rerank_enabled=False,
    )


@pytest.fixture
def container(settings: Settings) -> Iterator[Container]:
    built = Container.build(
        settings,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]  # stands in for Chroma
        embedder=FakeEmbedder(),
    )
    app.dependency_overrides[get_container] = lambda: built
    yield built
    app.dependency_overrides.clear()


@pytest.fixture
def document(tmp_path: Path) -> Path:
    path = tmp_path / "zebra.md"
    path.write_text("# Zebras\n\nThe quagga is an extinct zebra subspecies.\n")
    return path.resolve()


def _sources(result: Any) -> list[str]:
    return [hit["source"] for hit in result.structured_content["result"]]


async def test_api_ingest_is_visible_to_mcp_retrieval_in_the_same_process(
    container: Container, document: Path
) -> None:
    mcp = build_mcp_server(container.settings, lambda: container)

    async with Client(mcp) as client:
        before = await client.call_tool("search_documents", QUESTION)
        ingested = TestClient(app).post("/ingest", json={"path": str(document)})
        after = await client.call_tool("search_documents", QUESTION)

    assert (_sources(before), ingested.status_code, _sources(after)) == (
        [],
        200,
        [str(document)],
    )


async def test_api_rebuild_drops_a_removed_source_from_mcp_retrieval(
    container: Container, document: Path
) -> None:
    mcp = build_mcp_server(container.settings, lambda: container)
    client = TestClient(app)

    async with Client(mcp) as mcp_client:
        assert client.post("/ingest", json={"path": str(document)}).status_code == 200
        before = await mcp_client.call_tool("search_documents", QUESTION)
        await anyio.Path(document).unlink()
        rebuilt = client.post("/collections/rebuild", json={})
        after = await mcp_client.call_tool("search_documents", QUESTION)

    assert (_sources(before), rebuilt.status_code, _sources(after)) == (
        [str(document)],
        200,
        [],
    )


async def test_api_collection_delete_empties_mcp_retrieval(
    container: Container, document: Path
) -> None:
    mcp = build_mcp_server(container.settings, lambda: container)
    client = TestClient(app)

    async with Client(mcp) as mcp_client:
        assert client.post("/ingest", json={"path": str(document)}).status_code == 200
        deleted = client.delete("/collections/localrag")
        after = await mcp_client.call_tool("search_documents", QUESTION)

    assert (deleted.status_code, _sources(after)) == (200, [])


async def test_mcp_ingest_rebuilds_the_retriever(container: Container, document: Path) -> None:
    async with Client(build_mcp_server(container.settings, lambda: container)) as mcp_client:
        before = container.retriever
        await mcp_client.call_tool("ingest_path", {"path": str(document)})
        after = container.retriever

    assert after is not before


def test_async_api_ingest_clears_the_query_cache(settings: Settings, document: Path) -> None:
    query_cache = QueryCache(maxsize=4, ttl_seconds=60)
    query_cache.set("stale-answer", {"answer": "before the ingest"})
    assert query_cache.get("stale-answer") is not None
    container = Container.build(
        settings,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]  # stands in for Chroma
        embedder=FakeEmbedder(),
        query_cache=query_cache,
    )
    app.dependency_overrides[get_container] = lambda: container
    client = TestClient(app)

    job_id = client.post("/ingest/directory/async", json={"path": str(document.parent)}).json()[
        "job_id"
    ]
    deadline = time.monotonic() + 5.0
    while client.get(f"/ingest/jobs/{job_id}").json()["status"] != "done":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    app.dependency_overrides.clear()

    assert query_cache.get("stale-answer") is None


def test_every_consumer_shares_one_store_and_one_bm25_index(container: Container) -> None:
    def sharing() -> dict[str, bool]:
        store, bm25 = container.vector_store, container.bm25_index
        ingestion, retriever = container.ingestion_service, container.retriever
        return {
            "ingestion store": ingestion.vector_store is store,
            "ingestion bm25": ingestion.bm25_index is bm25,
            "retriever store": retriever.vector_store is store,
            "retriever bm25": retriever.bm25_index is bm25,
            "engine retriever": container.engine.retriever is retriever,
            "engine provider": container.engine.provider is container.llm_provider,
            "repository store": container.collection_repository._vector_store is store,  # noqa: SLF001 — the repository exposes no accessor
        }

    everything_shared = dict.fromkeys(sharing(), True)
    provider = container.llm_provider

    assert sharing() == everything_shared
    container.invalidate()
    assert (sharing(), container.llm_provider is provider) == (everything_shared, True)


def test_concurrent_first_use_builds_one_vector_store(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []

    def slow_create(cls: type[VectorStore], *, persist_path: str, collection_name: str) -> object:
        _ = (cls, collection_name)
        created.append(persist_path)
        time.sleep(0.05)
        return FakeVectorStore()

    monkeypatch.setattr(VectorStore, "create", classmethod(slow_create))
    container = Container.build(settings, embedder=FakeEmbedder())
    stores: list[object] = []
    threads = [
        threading.Thread(target=lambda: stores.append(container.vector_store)) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert (len(created), len({id(store) for store in stores})) == (1, 1)


def test_close_releases_what_the_container_built_and_spares_what_it_was_given(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_embedder = FakeEmbedder()
    given_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "localrag.application.container.build_embedding_provider", lambda _s: built_embedder
    )
    with Container.build(settings) as owning:
        _ = owning.embedder
    with Container.build(settings, embedder=given_embedder) as borrowing:
        _ = borrowing.embedder

    assert (built_embedder.closed, given_embedder.closed) == (True, False)


def test_plugin_registry_does_not_build_runtime_objects() -> None:
    source = Path(__file__).parents[1] / "localrag" / "plugins" / "retriever.py"
    imported = {
        node.module
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert {
        module
        for module in imported
        if module.startswith("localrag.application") or module == "localrag.rag.retriever"
    } == set()


def test_container_builds_its_collaborators_from_its_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        chroma_persist_path="persist",
        chroma_collection_name="collection-1",
        ollama_base_url="http://ollama:11434",
        ollama_embed_model="embed-model",
        ollama_llm_model="llm-model",
    )
    opened: list[tuple[str, str]] = []

    def create(cls: type[VectorStore], *, persist_path: str, collection_name: str) -> object:
        _ = cls
        opened.append((persist_path, collection_name))
        return FakeVectorStore()

    monkeypatch.setattr(VectorStore, "create", classmethod(create))
    container = Container.build(settings)
    _ = container.vector_store

    assert (
        opened,
        container.embedder.base_url,  # type: ignore[attr-defined]  # the Ollama provider
        container.embedder.model,
        container.retriever.settings,
        container.engine.settings,
        container.ingestion_service.settings,
    ) == (
        [("persist", "collection-1")],
        "http://ollama:11434",
        "embed-model",
        settings,
        settings,
        settings,
    )
