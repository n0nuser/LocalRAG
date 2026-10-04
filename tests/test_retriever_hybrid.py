from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from localrag.rag.bm25_index import Bm25Hit, Bm25Index
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
    def query(
        text: str, top_k: int, matches: Callable[[Mapping[str, Any]], bool] | None = None
    ) -> list[Bm25Hit]:
        _ = (text, top_k, matches)
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


@dataclass
class ChunkStore:
    """Serves one chunk list to both the dense path and the BM25 index."""

    chunks: list[tuple[str, str, dict[str, Any]]]

    def get_all_chunks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self.chunks

    def query(
        self, embedding: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, object]:
        _ = embedding
        rows = [
            (document, metadata)
            for _, document, metadata in self.chunks
            if all(metadata.get(key) == value for key, value in (where or {}).items())
        ][:top_k]
        return {
            "documents": [[document for document, _ in rows]],
            "metadatas": [[metadata for _, metadata in rows]],
            "distances": [[0.5] * len(rows)],
        }


def test_retriever_hybrid_ranks_filtered_lexical_hits_outranked_corpus_wide() -> None:
    # Every public chunk outranks every private one lexically, and the BM25 budget
    # (two per list here) is smaller than the public set, so only a filter applied
    # before top-k leaves any private chunk in the lexical list. With the lexical
    # list carrying all the weight, its order decides the result; without it every
    # candidate ties and the chunk_index tie-break puts the irrelevant chunk first.
    public = [
        (
            f"public-{n}",
            "lantern lantern lantern lantern",
            {"source": "public.md", "chunk_index": n},
        )
        for n in range(10)
    ]
    private = [
        ("private-a", "nothing relevant here", {"source": "private.md", "chunk_index": 9}),
        (
            "private-b",
            "a lantern by the door and a coat",
            {"source": "private.md", "chunk_index": 5},
        ),
        ("private-c", "the lantern was lit at dusk", {"source": "private.md", "chunk_index": 1}),
    ]
    store = ChunkStore(chunks=public + private)
    settings = Settings(
        retrieval_mode="hybrid",
        bm25_weight=1.0,
        freshness_weight=0.0,
        parent_expansion_enabled=False,
    )
    retriever = Retriever(
        settings=settings,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        bm25_index=Bm25Index.from_vector_store(store),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("lantern", n_results=2, metadata_filter={"source": "private.md"})

    assert [context["text"] for context in contexts] == [
        "the lantern was lit at dusk",
        "a lantern by the door and a coat",
    ]
