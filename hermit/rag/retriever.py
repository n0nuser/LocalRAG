from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermit.config import Settings
from hermit.ingestion.embedder import OllamaEmbedder
from hermit.storage.vector_store import VectorStore


@dataclass
class Retriever:
    settings: Settings
    embedder: OllamaEmbedder
    vector_store: VectorStore

    def retrieve(self, question: str, n_results: int | None = None) -> list[dict[str, Any]]:
        top_k = n_results if n_results is not None else self.settings.rag_top_k
        embedding = self.embedder.embed_text(question)
        query_result = self.vector_store.query(embedding=embedding, top_k=top_k)

        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        contexts: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            contexts.append(
                {
                    "text": document,
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index", -1),
                    "score": distance,
                }
            )
        return contexts
