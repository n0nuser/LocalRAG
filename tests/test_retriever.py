from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus

import httpx
import pytest
import respx

from localrag.ingestion.embedder import OllamaEmbedder
from localrag.rag.exceptions import RetrievalError
from localrag.rag.retriever import Retriever
from localrag.settings import Settings


@dataclass
class StubEmbedder:
    def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
        return [1.0, 2.0, 3.0]


@dataclass
class StubStore:
    def query(self, embedding: list[float], top_k: int) -> dict[str, object]:
        _ = (embedding, top_k)
        return {
            "documents": [["chunk-a"]],
            "metadatas": [[{"source": "foo.md", "chunk_index": 0}]],
            "distances": [[0.12]],
        }


def test_retriever_returns_contexts() -> None:
    settings = Settings()
    retriever = Retriever(
        settings=settings,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=StubStore(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("hello")

    assert contexts == [
        {
            "text": "chunk-a",
            "source": "foo.md",
            "chunk_index": 0,
            "score": pytest.approx(0.8928571428571428),
            "distance": 0.12,
            "ingested_at": None,
            "metadata": {"source": "foo.md", "chunk_index": 0},
            "freshness_factor": 1.0,
        }
    ]


@respx.mock
def test_retriever_raises_retrieval_failure_when_ollama_embed_fails() -> None:
    respx.post("http://ollama:11434/api/embed").mock(return_value=httpx.Response(503))
    embedder = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    retriever = Retriever(
        settings=Settings(),
        embedder=embedder,
        vector_store=StubStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(RetrievalError) as excinfo:
        retriever.retrieve("q")

    assert excinfo.value.status_code == HTTPStatus.BAD_GATEWAY


def test_retriever_raises_retrieval_failure_when_vector_query_fails() -> None:
    @dataclass
    class ExplodingStore:
        def query(self, embedding: list[float], top_k: int) -> dict[str, object]:
            raise RuntimeError("dimension mismatch")

    retriever = Retriever(
        settings=Settings(),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=ExplodingStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(RetrievalError) as excinfo:
        retriever.retrieve("q")

    assert excinfo.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE
