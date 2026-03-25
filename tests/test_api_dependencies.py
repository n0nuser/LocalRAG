from __future__ import annotations

import pytest

from localrag.api import dependencies as deps
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def list_collections(self) -> list[str]:
        return ["localrag"]

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)


def test_api_dependency_constructors_build_expected_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps.get_vector_store.cache_clear()
    deps.get_embedder.cache_clear()
    deps.get_retriever.cache_clear()
    deps.get_engine.cache_clear()
    deps.get_ingestion_service.cache_clear()

    settings = Settings(
        chroma_persist_path="persist",
        chroma_collection_name="collection-1",
        ollama_base_url="http://ollama:11434",
        ollama_embed_model="embed-model",
        ollama_llm_model="llm-model",
    )
    monkeypatch.setattr(deps, "get_settings", lambda: settings)

    created: list[tuple[str, str]] = []
    sentinel_vector_store = FakeVectorStore()

    def fake_create(cls: type[VectorStore], *, persist_path: str, collection_name: str) -> object:
        _ = cls
        created.append((persist_path, collection_name))
        return sentinel_vector_store

    monkeypatch.setattr(
        deps.VectorStore,
        "create",
        classmethod(fake_create),
    )

    store = deps.get_vector_store()
    assert store is sentinel_vector_store
    assert created == [(settings.chroma_persist_path, settings.chroma_collection_name)]

    embedder = deps.get_embedder()
    assert embedder.base_url == settings.ollama_base_url
    assert embedder.model == settings.ollama_embed_model

    retriever = deps.get_retriever()
    assert retriever.settings == settings
    assert retriever.vector_store is sentinel_vector_store

    engine = deps.get_engine()
    assert engine.settings == settings
    assert engine.retriever is retriever

    ingestion_service = deps.get_ingestion_service()
    assert ingestion_service.settings == settings
    assert ingestion_service.vector_store is sentinel_vector_store

    repo = deps.get_collection_repository(store=store)
    assert repo.list_collection_names() == ["localrag"]
    repo.delete_collection("x")
    assert sentinel_vector_store.deleted == ["x"]
