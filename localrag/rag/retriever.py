from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any

import httpx

from localrag.ingestion.embedder import OllamaEmbedder
from localrag.rag.bm25_index import Bm25Index
from localrag.rag.exceptions import RetrievalError
from localrag.rag.reranker import CrossEncoderReranker
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    return all(metadata.get(key) == value for key, value in metadata_filter.items())


@dataclass
class Retriever:
    settings: Settings
    embedder: OllamaEmbedder
    vector_store: VectorStore
    bm25_index: Bm25Index | None = None
    reranker: CrossEncoderReranker | None = None

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        top_k = n_results if n_results is not None else self.settings.rag_top_k
        fetch_k = (
            max(self.settings.rerank_fetch_k, top_k)
            if self.reranker is not None
            else max(top_k * 2, top_k)
        )
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

        vector_hits = self._retrieve_vector_hits(
            embedding=embedding, top_k=fetch_k, where=metadata_filter
        )
        if self.settings.retrieval_mode != "hybrid" or self.bm25_index is None:
            candidates = vector_hits
        else:
            bm25_hits = [
                {
                    "text": hit.text,
                    "source": hit.metadata.get("source", "unknown"),
                    "chunk_index": hit.metadata.get("chunk_index", -1),
                    "score": hit.score,
                    "ingested_at": hit.metadata.get("ingested_at"),
                    "metadata": hit.metadata,
                }
                for hit in self.bm25_index.query(question, top_k=fetch_k)
                if _matches_filter(hit.metadata, metadata_filter)
            ]
            candidates = self._fuse_results(
                vector_hits=vector_hits, bm25_hits=bm25_hits, top_k=fetch_k
            )

        if self.reranker is not None:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return self._expand_to_parent_section(self.apply_freshness(candidates))

    def _retrieve_vector_hits(
        self, embedding: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            query_result = self.vector_store.query(embedding=embedding, top_k=top_k, where=where)
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
            metadata_map = metadata if isinstance(metadata, dict) else {}
            contexts.append(
                {
                    "text": document,
                    "source": metadata_map.get("source", "unknown"),
                    "chunk_index": metadata_map.get("chunk_index", -1),
                    "score": 1.0 / (1.0 + float(distance)),
                    "distance": float(distance),
                    "ingested_at": metadata_map.get("ingested_at"),
                    "metadata": metadata_map,
                }
            )
        logger.debug("retrieve_vector_hits count=%s", len(contexts))
        return contexts

    def _fuse_results(
        self,
        *,
        vector_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        vector_sorted = sorted(vector_hits, key=lambda hit: float(hit["score"]), reverse=True)
        bm25_sorted = sorted(bm25_hits, key=lambda hit: float(hit["score"]), reverse=True)
        candidate_map: dict[tuple[str, int], dict[str, Any]] = {}
        score_map: dict[tuple[str, int], float] = {}
        rrf_k = max(1, self.settings.rrf_k)
        vector_weight = 1.0 - self.settings.bm25_weight
        bm25_weight = self.settings.bm25_weight

        for rank, hit in enumerate(vector_sorted, start=1):
            key = self._hit_key(hit)
            candidate_map[key] = hit
            score_map[key] = score_map.get(key, 0.0) + vector_weight / (rrf_k + rank)
        for rank, hit in enumerate(bm25_sorted, start=1):
            key = self._hit_key(hit)
            candidate_map[key] = hit
            score_map[key] = score_map.get(key, 0.0) + bm25_weight / (rrf_k + rank)

        ranked_keys = sorted(score_map.keys(), key=lambda key: score_map[key], reverse=True)[:top_k]
        fused: list[dict[str, Any]] = []
        for key in ranked_keys:
            hit = dict(candidate_map[key])
            hit["score"] = score_map[key]
            fused.append(hit)
        logger.debug("retrieve_hybrid_hits count=%s", len(fused))
        return fused

    def apply_freshness(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        half_life_days = self.settings.freshness_half_life_days
        if half_life_days <= 0:
            return [{**context, "freshness_factor": 1.0} for context in contexts]

        now = datetime.now(UTC)
        rescored: list[dict[str, Any]] = []
        for context in contexts:
            freshness_factor = 1.0
            ingested_at = context.get("ingested_at")
            if isinstance(ingested_at, str):
                try:
                    parsed = datetime.fromisoformat(ingested_at)
                except ValueError:
                    parsed = None
                if parsed is not None:
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
                    age_days = max(0.0, (now - parsed).total_seconds() / 86_400)
                    freshness_factor = 0.5 ** (age_days / half_life_days)
            rescored_context = dict(context)
            rescored_context["freshness_factor"] = freshness_factor
            rescored_context["score"] = float(context.get("score", 0.0)) * freshness_factor
            rescored.append(rescored_context)
        rescored.sort(key=lambda hit: float(hit.get("score", 0.0)), reverse=True)
        logger.debug("retrieve_hits count=%s", len(rescored))
        return rescored

    @staticmethod
    def _hit_key(hit: dict[str, Any]) -> tuple[str, int]:
        source = str(hit.get("source", "unknown"))
        chunk_index = int(hit.get("chunk_index", -1))
        return source, chunk_index

    def _expand_to_parent_section(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.settings.parent_expansion_enabled:
            return contexts
        expanded: list[dict[str, Any]] = []
        for context in contexts:
            metadata = context.get("metadata") or {}
            heading_path = metadata.get("heading_path", "")
            source = context.get("source", "unknown")
            if not heading_path:
                expanded.append(context)
                continue
            siblings = self.vector_store.get_chunks_by_heading(
                source=str(source), heading_path=str(heading_path)
            )
            if len(siblings) <= 1:
                expanded.append(context)
                continue
            merged_text = "\n\n".join(text for _, text in siblings)
            new_context = dict(context)
            new_context["expanded_text"] = merged_text
            expanded.append(new_context)
        return expanded
