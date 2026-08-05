from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any

import httpx

from localrag.embedding.base import EmbeddingError, EmbeddingProvider
from localrag.observability.tracing import SpanName, span
from localrag.rag.bm25_index import Bm25Index
from localrag.rag.exceptions import RetrievalError
from localrag.rag.hyde import HydeObservation, generate_hypothetical
from localrag.rag.query_rewrite import expand_query, rewrite_query
from localrag.rag.reranker import CrossEncoderReranker
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_HARD_MAX_CANDIDATES = 100


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


@dataclass(frozen=True)
class _QueryPlan:
    """Resolved budgets and query set for a single retrieval, fixed before any I/O."""

    top_k: int
    per_variant_k: int
    search_question: str
    lexical_question: str
    variants: tuple[str, ...]
    hypothetical: str | None

    @property
    def dense_queries(self) -> tuple[str, ...]:
        """A HyDE passage replaces the variants for dense search when present."""
        return (self.hypothetical,) if self.hypothetical else self.variants


@dataclass
class Retriever:
    settings: Settings
    embedder: EmbeddingProvider
    vector_store: VectorStore
    bm25_index: Bm25Index | None = None
    reranker: CrossEncoderReranker | None = None
    last_hyde: HydeObservation | None = None

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the retrieval stages in order: plan, dense, fuse, rerank, post-process."""
        plan = self._plan_query(question, n_results)
        vector_lists = self._dense_retrieve(plan, metadata_filter)
        candidates, fused = self._fuse_candidates(plan, vector_lists, metadata_filter)
        candidates = self._rerank(question, candidates, plan.top_k)
        # Hybrid fusion already accounts for recency as its own ranked list, so the
        # multiplicative decay would double-count it. In vector-only mode scores
        # spread widely enough for decay to act as the intended tiebreaker.
        with span(SpanName.RETRIEVAL_FRESHNESS, {"count": len(candidates)}):
            return self._expand_to_parent_section(
                self.apply_freshness(candidates, rescore=not fused), metadata_filter
            )

    def _plan_query(self, question: str, n_results: int | None) -> _QueryPlan:
        """Resolve budgets and the rewrite/HyDE/expansion query set for one retrieval."""
        top_k = n_results if n_results is not None else self.settings.rag_top_k
        fetch_k = (
            max(self.settings.rerank_fetch_k, top_k)
            if self.reranker is not None
            else max(top_k * 2, top_k)
        )
        mode = self.settings.retrieval_experiment_mode
        rewrite_enabled = self.settings.query_rewrite_enabled
        hyde_enabled = self.settings.hyde_enabled
        if mode != "auto":
            rewrite_enabled = mode in {"rewrite", "rewrite+hyde"}
            hyde_enabled = mode in {"hyde", "rewrite+hyde"}

        search_question = question
        rewritten: str | None = None
        if rewrite_enabled:
            search_question = rewrite_query(question, self.settings)
            rewritten = search_question if search_question != question else None
            logger.debug(
                "query_rewritten original_chars=%s rewritten_chars=%s",
                len(question),
                len(search_question),
            )

        hyde_settings = self.settings.model_copy(update={"hyde_enabled": hyde_enabled})
        hypothetical, self.last_hyde = generate_hypothetical(search_question, hyde_settings)
        lexical_question = question
        if self.settings.hyde_lexical_input == "rewritten" and rewritten is not None:
            lexical_question = search_question

        expansion = expand_query(question, search_question, self.settings, rewrite=rewritten)
        variants = expansion.variants
        candidate_budget = (
            min(self.settings.query_expansion_candidate_budget, _HARD_MAX_CANDIDATES)
            if self.settings.query_expansion_enabled
            else fetch_k
        )
        rank_list_count = 2 if self.settings.retrieval_mode == "hybrid" and self.bm25_index else 1
        per_variant_k = min(fetch_k, max(1, candidate_budget // (len(variants) * rank_list_count)))
        logger.debug(
            "retrieve_query_plan status=%s variants=%s candidate_budget=%s",
            expansion.status,
            len(variants),
            candidate_budget if self.settings.query_expansion_enabled else fetch_k,
        )
        return _QueryPlan(
            top_k=top_k,
            per_variant_k=per_variant_k,
            search_question=search_question,
            lexical_question=lexical_question,
            variants=variants,
            hypothetical=hypothetical,
        )

    def _dense_retrieve(
        self, plan: _QueryPlan, metadata_filter: dict[str, Any] | None
    ) -> list[list[dict[str, Any]]]:
        """Embed each dense query and collect one ranked list per variant."""
        vector_lists: list[list[dict[str, Any]]] = []
        try:
            ensure = getattr(self.vector_store, "ensure_embedding_compatibility", None)
            if ensure is not None:
                ensure(self.embedder)
            for variant in plan.dense_queries:
                embedding = self._embed_variant(variant, plan)
                if ensure is not None:
                    ensure(self.embedder, len(embedding))
                with span(SpanName.RETRIEVAL_VECTOR, {"count": plan.per_variant_k}):
                    vector_lists.append(
                        self._retrieve_vector_hits(embedding, plan.per_variant_k, metadata_filter)
                    )
        except (httpx.HTTPError, EmbeddingError) as exc:
            logger.error(
                "retrieve_embed_provider_error provider=%s model=%s error=%s",
                getattr(self.embedder, "provider_name", "embedding"),
                getattr(self.embedder, "model", "unknown"),
                exc,
            )
            raise RetrievalError(
                HTTPStatus.BAD_GATEWAY,
                "Embedding service unavailable.",
            ) from exc
        return vector_lists

    def _embed_variant(self, variant: str, plan: _QueryPlan) -> list[float]:
        """Embed one query, falling back off a failed hypothetical rather than failing."""
        embed = getattr(self.embedder, "embed", None)
        legacy_embedder: Any = self.embedder

        def _run(text: str) -> list[float]:
            return embed(text) if embed is not None else legacy_embedder.embed_text(text)

        try:
            return _run(variant)
        except (httpx.HTTPError, EmbeddingError):
            # Only a HyDE passage is recoverable: retry with the real query instead.
            if plan.hypothetical is None or variant != plan.hypothetical:
                raise
            embedding = _run(plan.search_question)
            if self.last_hyde is not None:
                self.last_hyde = replace(
                    self.last_hyde,
                    mode="fallback",
                    status="fallback",
                    fallback_reason="embedding_failure",
                )
            return embedding

    def _fuse_candidates(
        self,
        plan: _QueryPlan,
        vector_lists: list[list[dict[str, Any]]],
        metadata_filter: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Combine dense (and, in hybrid mode, lexical) lists into one ranked list."""
        if self.settings.retrieval_mode != "hybrid" or self.bm25_index is None:
            if len(vector_lists) == 1:
                return vector_lists[0], False
            return (
                self._fuse_rank_lists(
                    vector_lists,
                    [1.0 / len(vector_lists)] * len(vector_lists),
                    plan.per_variant_k,
                ),
                False,
            )

        bm25_queries = plan.variants if not plan.hypothetical else (plan.lexical_question,)
        bm25_lists = [
            self._bm25_hits(variant, plan.per_variant_k, metadata_filter)
            for variant in bm25_queries
        ]
        lists = vector_lists + bm25_lists
        dense_count = len(plan.dense_queries)
        bm25_count = len(bm25_queries)
        weights = [(1.0 - self.settings.bm25_weight) / dense_count] * dense_count + [
            self.settings.bm25_weight / bm25_count
        ] * bm25_count
        with span(SpanName.RETRIEVAL_RRF, {"count": len(lists)}):
            return self._fuse_rank_lists(lists, weights, plan.per_variant_k), True

    def _bm25_hits(
        self, variant: str, per_variant_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if self.bm25_index is None:
            return []
        with span(SpanName.RETRIEVAL_BM25, {"count": per_variant_k}):
            return [
                {
                    "text": hit.text,
                    "chunk_id": hit.chunk_id,
                    "source": hit.metadata.get("source", "unknown"),
                    "chunk_index": hit.metadata.get("chunk_index", -1),
                    "score": hit.score,
                    "ingested_at": hit.metadata.get("ingested_at"),
                    "metadata": hit.metadata,
                }
                for hit in self.bm25_index.query(variant, top_k=per_variant_k)
                if _matches_filter(hit.metadata, metadata_filter)
            ]

    def _rerank(
        self, question: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if self.reranker is None:
            return candidates[:top_k]
        with span(SpanName.RETRIEVAL_RERANK, {"count": len(candidates)}):
            return self.reranker.rerank(question, candidates, top_k=top_k)

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
            context = {
                "text": document,
                "source": metadata_map.get("source", "unknown"),
                "chunk_index": metadata_map.get("chunk_index", -1),
                "score": 1.0 / (1.0 + float(distance)),
                "distance": float(distance),
                "ingested_at": metadata_map.get("ingested_at"),
                "metadata": metadata_map,
            }
            if metadata_map.get("chunk_id"):
                context["chunk_id"] = metadata_map["chunk_id"]
            contexts.append(context)
        logger.debug("retrieve_vector_hits count=%s", len(contexts))
        return contexts

    def _fuse_results(
        self,
        *,
        vector_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return self._fuse_rank_lists(
            [vector_hits, bm25_hits],
            [1.0 - self.settings.bm25_weight, self.settings.bm25_weight],
            top_k,
        )

    def _fuse_rank_lists(
        self,
        rank_lists: list[list[dict[str, Any]]],
        weights: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse explicit ranked lists, deduplicating by source and chunk index."""
        # Ties are broken by recency and stable identity so equally-relevant candidates
        # do not depend on provider/list insertion order.
        if len(rank_lists) != len(weights):
            raise ValueError("rank_lists and weights must have the same length")
        # arbitrary order from the sort's stability — that order would otherwise
        # decide the fused ranking, since a tied pair differs by a single rank in
        # every list.
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
        for hits, weight in zip(rank_lists, weights, strict=True):
            for rank, hit in enumerate(self._sorted_by_score(hits), start=1):
                key = self._hit_key(hit)
                candidate_map.setdefault(key, hit)
                score_map[key] = score_map.get(key, 0.0) + relevance_weight * weight / (
                    rrf_k + rank
                )
        if freshness_weight > 0.0:
            for key, rank in self._recency_ranks(candidate_map).items():
                score_map[key] = score_map.get(key, 0.0) + freshness_weight / (rrf_k + rank)

        ranked_keys = sorted(
            score_map,
            key=lambda key: (
                score_map[key],
                _parse_ingested_at(candidate_map[key].get("ingested_at"))
                or datetime.min.replace(tzinfo=UTC),
                key,
            ),
            reverse=True,
        )[:top_k]
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

    def _expand_to_parent_section(
        self,
        contexts: list[dict[str, Any]],
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.settings.parent_expansion_enabled:
            return contexts
        parent_keys = {
            (
                str(context.get("source", "unknown")),
                str((context.get("metadata") or {}).get("heading_path")),
            )
            for context in contexts
            if (context.get("metadata") or {}).get("heading_path")
        }
        if not parent_keys:
            return contexts
        bulk_lookup = getattr(self.vector_store, "get_chunks_by_headings", None)
        sections = bulk_lookup(list(parent_keys), metadata_filter) if bulk_lookup else {}
        expanded: list[dict[str, Any]] = []
        for context in contexts:
            metadata = context.get("metadata") or {}
            heading_path = metadata.get("heading_path", "")
            source = context.get("source", "unknown")
            if not heading_path:
                expanded.append(context)
                continue
            siblings = sections.get((str(source), str(heading_path)), [])
            if len(siblings) <= 1:
                expanded.append(context)
                continue
            merged_text = "\n\n".join(text for _, text in siblings)
            new_context = dict(context)
            new_context["expanded_text"] = merged_text
            expanded.append(new_context)
        return expanded
