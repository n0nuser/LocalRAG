from __future__ import annotations

from dataclasses import dataclass

from localrag.rag.bm25_index import Bm25Hit
from localrag.rag.retriever import Retriever
from localrag.settings import Settings


@dataclass
class StubEmbedder:
    @staticmethod
    def embed_text(text: str, *, model: str | None = None) -> list[float]:
        _ = (text, model)
        return [0.1, 0.2, 0.3]


@dataclass
class StubStore:
    @staticmethod
    def query(
        embedding: list[float], top_k: int, where: dict[str, object] | None = None
    ) -> dict[str, object]:
        _ = (embedding, top_k, where)
        return {
            "documents": [["nearby vector text", "exact token text"]],
            "metadatas": [
                [
                    {"source": "nearby.md", "chunk_index": 0},
                    {"source": "exact.md", "chunk_index": 1},
                ]
            ],
            "distances": [[0.01, 0.4]],
        }


@dataclass
class StubBm25Index:
    @staticmethod
    def query(text: str, top_k: int) -> list[Bm25Hit]:
        _ = (text, top_k)
        return [
            Bm25Hit(
                chunk_id="exact",
                text="exact token text",
                metadata={"source": "exact.md", "chunk_index": 1},
                score=8.0,
            )
        ]


def test_retriever_hybrid_fuses_vector_and_bm25() -> None:
    settings = Settings(retrieval_mode="hybrid", rrf_k=1)
    retriever = Retriever(
        settings=settings,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=StubStore(),  # type: ignore[arg-type]
        bm25_index=StubBm25Index(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("ERR_QUIC_PROTOCOL_ERROR", n_results=2)

    assert contexts[0]["source"] == "exact.md"


def test_retriever_hybrid_applies_metadata_filter_to_bm25_hits() -> None:
    @dataclass
    class FilteringStore:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k)
            documents = ["nearby vector text", "exact token text"]
            metadatas = [
                {"source": "nearby.md", "chunk_index": 0},
                {"source": "exact.md", "chunk_index": 1},
            ]
            if where:
                keep = [
                    i
                    for i, md in enumerate(metadatas)
                    if all(md.get(k) == v for k, v in where.items())
                ]
                documents = [documents[i] for i in keep]
                metadatas = [metadatas[i] for i in keep]
            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [[0.01] * len(documents)],
            }

    settings = Settings(retrieval_mode="hybrid", rrf_k=1)
    retriever = Retriever(
        settings=settings,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=FilteringStore(),  # type: ignore[arg-type]
        bm25_index=StubBm25Index(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve(
        "ERR_QUIC_PROTOCOL_ERROR", n_results=2, metadata_filter={"source": "nearby.md"}
    )

    assert all(context["source"] == "nearby.md" for context in contexts)
