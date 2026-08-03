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
from localrag.rag.query_rewrite import rewrite_query
from localrag.rag.reranker import CrossEncoderReranker
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _parse_ingested_at(value: Any) -> datetime | None:
    """Parse an ``ingested_at`` metadata value, returning None when unusable."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


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
        search_question = question
        if self.settings.query_rewrite_enabled:
            search_question = rewrite_query(question, self.settings)
            logger.debug(
                "query_rewritten original_chars=%s rewritten_chars=%s",
                len(question),
                len(search_question),
            )
        logger.debug(
            "retrieve_embed_question top_k=%s question_chars=%s", top_k, len(search_question)
        )
        try:
            embedding = self.embedder.embed_text(search_question)
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
        fused = False
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
                for hit in self.bm25_index.query(search_question, top_k=fetch_k)
                if _matches_filter(hit.metadata, metadata_filter)
            ]
            candidates = self._fuse_results(
                vector_hits=vector_hits, bm25_hits=bm25_hits, top_k=fetch_k
            )
            fused = True

        if self.reranker is not None:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Hybrid fusion already accounts for recency as its own ranked list, so the
        # multiplicative decay would double-count it. In vector-only mode scores
        # spread widely enough for decay to act as the intended tiebreaker.
        return self._expand_to_parent_section(self.apply_freshness(candidates, rescore=not fused))

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
        # Ties are broken by recency so equally-relevant candidates do not take an
        # arbitrary order from the sort's stability — that order would otherwise
        # decide the fused ranking, since a tied pair differs by a single rank in
        # every list.
        vector_sorted = self._sorted_by_score(vector_hits)
        bm25_sorted = self._sorted_by_score(bm25_hits)
        candidate_map: dict[tuple[str, int], dict[str, Any]] = {}
        score_map: dict[tuple[str, int], float] = {}
        rrf_k = max(1, self.settings.rrf_k)
        # Recency participates as a third ranked list rather than as a multiplier on
        # the fused score. RRF scores are deliberately compressed — adjacent ranks
        # differ by well under 2% at rrf_k=60 — so any multiplicative factor with a
        # wider range than that reorders results by that factor instead of by
        # relevance. Contributing a rank term keeps recency bounded by its weight.
        freshness_weight = max(0.0, min(1.0, self.settings.freshness_weight))
        if self.settings.freshness_half_life_days <= 0:
            freshness_weight = 0.0
        relevance_weight = 1.0 - freshness_weight
        vector_weight = relevance_weight * (1.0 - self.settings.bm25_weight)
        bm25_weight = relevance_weight * self.settings.bm25_weight

        for rank, hit in enumerate(vector_sorted, start=1):
            key = self._hit_key(hit)
            candidate_map[key] = hit
            score_map[key] = score_map.get(key, 0.0) + vector_weight / (rrf_k + rank)
        for rank, hit in enumerate(bm25_sorted, start=1):
            key = self._hit_key(hit)
            candidate_map[key] = hit
            score_map[key] = score_map.get(key, 0.0) + bm25_weight / (rrf_k + rank)
        if freshness_weight > 0.0:
            for key, rank in self._recency_ranks(candidate_map).items():
                score_map[key] = score_map.get(key, 0.0) + freshness_weight / (rrf_k + rank)

        ranked_keys = sorted(score_map.keys(), key=lambda key: score_map[key], reverse=True)[:top_k]
        fused: list[dict[str, Any]] = []
        for key in ranked_keys:
            hit = dict(candidate_map[key])
            hit["score"] = score_map[key]
            fused.append(hit)
        logger.debug("retrieve_hybrid_hits count=%s", len(fused))
        return fused

    @staticmethod
    def _sorted_by_score(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort hits by score descending, breaking ties by recency."""
        epoch = datetime.min.replace(tzinfo=UTC)
        return sorted(
            hits,
            key=lambda hit: (
                float(hit["score"]),
                _parse_ingested_at(hit.get("ingested_at")) or epoch,
            ),
            reverse=True,
        )

    @staticmethod
    def _recency_ranks(
        candidate_map: dict[tuple[str, int], dict[str, Any]],
    ) -> dict[tuple[str, int], int]:
        """Rank candidate keys newest-first, giving undated candidates a neutral rank.

        A candidate with no usable ``ingested_at`` takes the middle rank of the
        dated ones rather than being ranked last or dropped: missing metadata
        should neither reward nor penalise a document, and dropping it would
        forfeit the recency weight entirely, which is itself a penalty.
        """
        dated: list[tuple[datetime, tuple[str, int]]] = []
        undated: list[tuple[str, int]] = []
        for key, hit in candidate_map.items():
            parsed = _parse_ingested_at(hit.get("ingested_at"))
            if parsed is None:
                undated.append(key)
            else:
                dated.append((parsed, key))
        dated.sort(key=lambda item: item[0], reverse=True)

        ranks = {key: rank for rank, (_, key) in enumerate(dated, start=1)}
        neutral_rank = (len(dated) + 1) // 2 or 1
        for key in undated:
            ranks[key] = neutral_rank
        return ranks

    def apply_freshness(
        self, contexts: list[dict[str, Any]], *, rescore: bool = True
    ) -> list[dict[str, Any]]:
        """Attach ``freshness_factor`` to each context, optionally rescoring by it.

        Args:
            contexts: Retrieved candidates, already ranked.
            rescore: When true, multiply each score by its freshness factor and
                re-sort. Hybrid retrieval passes false: recency is already folded
                into the fused score as its own ranked list, and multiplying again
                would let decay dominate the compressed RRF score range.

        Returns:
            The contexts with ``freshness_factor`` populated for observability.
        """
        half_life_days = self.settings.freshness_half_life_days
        if half_life_days <= 0:
            return [{**context, "freshness_factor": 1.0} for context in contexts]

        now = datetime.now(UTC)
        rescored: list[dict[str, Any]] = []
        for context in contexts:
            freshness_factor = 1.0
            parsed = _parse_ingested_at(context.get("ingested_at"))
            if parsed is not None:
                age_days = max(0.0, (now - parsed).total_seconds() / 86_400)
                freshness_factor = 0.5 ** (age_days / half_life_days)
            rescored_context = dict(context)
            rescored_context["freshness_factor"] = freshness_factor
            if rescore:
                rescored_context["score"] = float(context.get("score", 0.0)) * freshness_factor
            rescored.append(rescored_context)
        if rescore:
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
