from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hermit.ingestion.embedder import OllamaEmbedder
from hermit.settings import Settings
from hermit.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Retriever:
    settings: Settings
    embedder: OllamaEmbedder
    vector_store: VectorStore

    def retrieve(self, question: str, n_results: int | None = None) -> list[dict[str, Any]]:
        top_k = n_results if n_results is not None else self.settings.rag_top_k
        logger.debug("retrieve_embed_question top_k=%s question_chars=%s", top_k, len(question))
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
        logger.debug("retrieve_hits count=%s", len(contexts))
        return contexts
