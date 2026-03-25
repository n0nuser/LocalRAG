from __future__ import annotations

from dataclasses import dataclass

from localrag.rag.retriever import Retriever
from localrag.settings import Settings


@dataclass
class StubEmbedder:
    def embed_text(self, text: str) -> list[float]:
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
        embedder=StubEmbedder(),
        vector_store=StubStore(),
    )

    contexts = retriever.retrieve("hello")

    assert contexts == [{"text": "chunk-a", "source": "foo.md", "chunk_index": 0, "score": 0.12}]
