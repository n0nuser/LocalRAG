from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx
import pytest
import respx

from localrag.embedding.base import (
    EmbeddingConfigurationError,
    EmbeddingIncompatibilityError,
    EmbeddingProvider,
)
from localrag.embedding.factory import build_embedding_provider
from localrag.ingestion.embedder import OllamaEmbedder
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore


@dataclass
class DeterministicProvider:
    provider_name: str = "deterministic"
    model: str = "test-v1"
    dimension: int | None = 2
    timeout_seconds: float = 1

    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        return self.embed_batch([text], model=model)[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        model: str | None = None,
    ) -> list[list[float]]:
        _ = (batch_size, model)
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    def close(self) -> None:
        return None


@pytest.mark.parametrize("batch_size", [1, 2, 10, None])
def test_provider_contract_preserves_order_and_empty_input(batch_size: int | None) -> None:
    provider: EmbeddingProvider = DeterministicProvider()

    assert provider.embed_batch([], batch_size=batch_size) == []
    assert provider.embed_batch(["a", "long"], batch_size=batch_size) == [
        [1.0, 0.0],
        [4.0, 1.0],
    ]
    assert provider.embed("a") == [1.0, 0.0]


@respx.mock
@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_ollama_provider_contract_batch_boundaries(batch_size: int) -> None:
    route = respx.post("http://ollama:11434/api/embed").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[float(len(item))] for item in json.loads(request.content)["input"]],
            },
        ),
    )
    provider = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    assert provider.embed_batch(["a", "bb", "ccc"], batch_size=batch_size) == [[1.0], [2.0], [3.0]]
    assert len(route.calls) == (3 if batch_size == 1 else 2 if batch_size == 2 else 1)
    assert provider.embed_batch([]) == []


def test_factory_preserves_legacy_ollama_model_and_supports_selector() -> None:
    provider = build_embedding_provider(
        Settings(ollama_embed_model="legacy-model", embedding_model="")
    )
    assert isinstance(provider, OllamaEmbedder)
    assert provider.model == "legacy-model"

    with pytest.raises(EmbeddingConfigurationError, match="Unsupported"):
        build_embedding_provider(Settings(embedding_provider="unknown"))


@dataclass
class FakeCollection:
    metadata: dict[str, object] = field(
        default_factory=lambda: {
            "hnsw:space": "cosine",
            "localrag:embedding_provider": "ollama",
            "localrag:embedding_model": "old",
            "localrag:embedding_dimension": 3,
        }
    )
    stored_embeddings: list[list[float]] = field(default_factory=list)

    def modify(self, *, metadata: dict[str, object]) -> None:
        self.metadata = metadata

    def get(self, *, include: list[str], limit: int) -> dict[str, object]:
        return {"embeddings": self.stored_embeddings[:limit] if "embeddings" in include else []}


def test_collection_rejects_incompatible_embedding_space() -> None:
    provider = DeterministicProvider()
    store = VectorStore(client=object(), collection=FakeCollection())  # type: ignore[arg-type]

    with pytest.raises(EmbeddingIncompatibilityError, match="rebuild"):
        store.ensure_embedding_compatibility(provider)


def test_collection_records_embedding_identity_after_validation() -> None:
    provider = DeterministicProvider()
    collection = FakeCollection()
    collection.metadata = {"hnsw:space": "cosine"}
    store = VectorStore(client=object(), collection=collection)  # type: ignore[arg-type]

    store.record_embedding_compatibility(provider, 2)

    assert collection.metadata["localrag:embedding_provider"] == "deterministic"
    assert collection.metadata["localrag:embedding_dimension"] == 2


def test_legacy_collection_dimension_is_checked_before_adoption() -> None:
    collection = FakeCollection(
        metadata={"hnsw:space": "cosine"}, stored_embeddings=[[1.0, 2.0, 3.0]]
    )
    store = VectorStore(client=object(), collection=collection)  # type: ignore[arg-type]

    with pytest.raises(EmbeddingIncompatibilityError, match="legacy collection"):
        store.ensure_embedding_compatibility(DeterministicProvider())
