from __future__ import annotations

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx

from localrag.ingestion.embedder import OllamaEmbedder
from localrag.rag.exceptions import RetrievalError
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Retriever:
    settings: Settings
    embedder: OllamaEmbedder
    vector_store: VectorStore

    def retrieve(self, question: str, n_results: int | None = None) -> list[dict[str, Any]]:
        top_k = n_results if n_results is not None else self.settings.rag_top_k
        logger.debug("retrieve_embed_question top_k=%s question_chars=%s", top_k, len(question))
        try:
            embedding = self.embedder.embed_text(question)
        except httpx.HTTPError as exc:
            logger.error(
                "retrieve_embed_ollama_http_error url=%s error=%s",
                self.embedder.base_url,
                exc,
            )
            raise RetrievalError(
                HTTPStatus.BAD_GATEWAY,
                "Embedding service unavailable.",
            ) from exc
        except ValueError as exc:
            logger.error("retrieve_embed_invalid_response error=%s", exc)
            raise RetrievalError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc

        try:
            query_result = self.vector_store.query(embedding=embedding, top_k=top_k)
        except Exception as exc:
            logger.exception("retrieve_vector_store_query_failed top_k=%s", top_k)
            raise RetrievalError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Vector store query failed. The collection may be inconsistent "
                "(for example embedding dimension mismatch). Try rebuilding the collection.",
            ) from exc

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
