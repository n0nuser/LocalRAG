from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from localrag.llm.providers.base import BaseLLMProvider
from localrag.observability.tracing import SpanName, span
from localrag.rag.adaptive import AdaptiveRetrievalPolicy
from localrag.rag.claim_filter import (
    ClaimFilterObservation,
    ClaimFilterStatus,
    filter_inapplicable_contexts,
)
from localrag.rag.compressor import CompressionBudget, compress_contexts
from localrag.rag.prompt import build_prompt
from localrag.rag.retriever import Retriever
from localrag.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class RAGEngine:
    settings: Settings
    retriever: Retriever
    provider: BaseLLMProvider

    def for_collection(self, collection: str) -> RAGEngine:
        """Create a request-scoped engine targeting one collection."""
        if not hasattr(self.retriever, "for_collection"):
            raise TypeError("Per-request collections require the built-in retriever.")
        return type(self)(
            settings=self.settings.with_overrides(chroma_collection_name=collection),
            retriever=self.retriever.for_collection(collection),
            provider=self.provider,
        )

    def answer(
        self,
        question: str,
        model: str | None = None,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        if self.settings.adaptive_enabled:
            events = list(self.stream_answer(question, model, n_results, metadata_filter))
            return {
                "answer": "".join(
                    str(event["token"]) for event in events if event["type"] == "token"
                ).strip(),
                "sources": next(
                    (list(event["sources"]) for event in events if event["type"] == "final"), []
                ),
                "trace": next(
                    (event.get("trace") for event in events if event["type"] == "final"), None
                ),
                "retrieved_chunks": next(
                    (
                        event.get("retrieved_chunks", 0)
                        for event in events
                        if event["type"] == "final"
                    ),
                    0,
                ),
            }
        chunks: list[str] = []
        sources: list[dict[str, object]] = []
        for event in self.stream_answer(
            question=question, model=model, n_results=n_results, metadata_filter=metadata_filter
        ):
            if event["type"] == "token":
                chunks.append(str(event["token"]))
            if event["type"] == "final":
                sources = list(event["sources"])
        return {"answer": "".join(chunks).strip(), "sources": sources}

    def stream_answer(
        self,
        question: str,
        model: str | None = None,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any]]:
        logger.info(
            "rag_stream_start question_chars=%s model=%s n_results=%s",
            len(question),
            model,
            n_results,
        )
        if self.settings.adaptive_enabled:
            with span(SpanName.RETRIEVAL_ADAPTIVE, {"stage": "adaptive"}):
                result = AdaptiveRetrievalPolicy(self.settings, self.retriever, self.provider).run(
                    question, model=model, n_results=n_results, metadata_filter=metadata_filter
                )
            if result.trace.abstained:
                yield from self._adaptive_refusal(result.trace)
                return
            contexts = result.contexts
            stream = self.stream_chat_from_contexts(
                contexts=contexts, question=question, model=model
            )
            for event in stream:
                if event["type"] == "final":
                    trace = result.trace.to_dict()
                    trace["hyde"] = _hyde_trace(getattr(self.retriever, "last_hyde", None))
                    event["trace"] = trace
                    event["retrieved_chunks"] = len(contexts)
                yield event
            return
        with span(SpanName.RETRIEVAL, {"stage": "retrieve"}):
            contexts = self.retriever.retrieve(
                question=question, n_results=n_results, metadata_filter=metadata_filter
            )
        yield from self.stream_chat_from_contexts(contexts=contexts, question=question, model=model)

    def stream_chat_from_contexts(
        self,
        *,
        contexts: list[dict[str, Any]],
        question: str,
        model: str | None,
    ) -> Generator[dict[str, Any]]:
        """Stream LLM tokens when contexts were retrieved earlier (HTTP runs retrieve first)."""
        if self._is_low_confidence(contexts):
            logger.info("rag_low_confidence_refusal question_chars=%s", len(question))
            return self._low_confidence_response(getattr(self.retriever, "last_hyde", None))
        return self._stream_chat_tokens(contexts=contexts, question=question, model=model)

    def _is_low_confidence(self, contexts: list[dict[str, Any]]) -> bool:
        min_score = self.settings.rag_min_context_score
        if min_score <= 0:
            return False
        if not contexts:
            return True
        top_score = float(contexts[0].get("score", 0.0))
        return top_score < min_score

    @staticmethod
    def _low_confidence_response(trace: Any = None) -> Generator[dict[str, Any]]:
        yield {
            "type": "token",
            "token": "I don't have enough information in the ingested documents to answer that.",
        }
        yield {"type": "final", "sources": [], "low_confidence": True, "trace": _hyde_trace(trace)}

    @staticmethod
    def _adaptive_refusal(trace: Any) -> Generator[dict[str, Any]]:
        yield {
            "type": "token",
            "token": "I don't have enough information in the ingested documents to answer that.",
        }
        yield {"type": "final", "sources": [], "low_confidence": True, "trace": trace.to_dict()}

    def _stream_chat_tokens(
        self,
        *,
        contexts: list[dict[str, Any]],
        question: str,
        model: str | None,
    ) -> Generator[dict[str, Any]]:
        logger.debug("rag_contexts count=%s", len(contexts))
        # Applicability filtering runs before compression so the compression budget is
        # spent only on passages that can actually answer the question at its scope.
        claim_filter = filter_inapplicable_contexts(
            contexts, question, self.settings, self.provider
        )
        # Compression is unconditional: it is deterministic, extractive, and needs no
        # provider call, so its budgets alone decide how much context survives.
        with span(SpanName.RETRIEVAL_COMPRESSION, {"count": len(claim_filter.contexts)}):
            compression = compress_contexts(
                claim_filter.contexts,
                question,
                CompressionBudget(
                    max_contexts=self.settings.context_compression_max_contexts,
                    candidate_count=self.settings.context_compression_candidate_count,
                    per_context_tokens=self.settings.context_compression_per_context_tokens,
                    total_tokens=self.settings.context_compression_total_tokens,
                    per_context_chars=self.settings.context_compression_per_context_chars,
                    total_chars=self.settings.context_compression_total_chars,
                ),
            )
        prompt_contexts = compression.contexts
        logger.info(
            "rag_context_compression status=%s input_tokens=%s output_tokens=%s",
            compression.status,
            compression.input_tokens,
            compression.output_tokens,
        )
        prompt = build_prompt(
            system_prompt=self.settings.rag_system_prompt,
            question=question,
            contexts=prompt_contexts,
        )
        with span(SpanName.GENERATION, {"model": model or "default"}):
            for event in self.provider.stream_from_prompt(prompt, model=model):
                if event["type"] == "token":
                    yield event
        logger.info("rag_stream_done")
        yield {
            "type": "final",
            # Sources come from the filtered set: a passage the filter discarded did
            # not inform the answer, so citing it would misattribute the response.
            "sources": self.extract_sources(claim_filter.contexts),
            "low_confidence": False,
            "trace": _merge_trace(
                _hyde_trace(getattr(self.retriever, "last_hyde", None)),
                claim_filter.observation,
            ),
        }

    @staticmethod
    def extract_sources(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, int]] = set()
        sources: list[dict[str, Any]] = []
        for context in contexts:
            source = str(context.get("source", "unknown"))
            chunk_index = int(context.get("chunk_index", -1))
            key = (source, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            metadata = context.get("metadata") or {}
            sources.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "heading_path": metadata.get("heading_path") or None,
                    "chunk_type": metadata.get("chunk_type") or None,
                }
            )
        return sources


def _merge_trace(
    hyde: dict[str, Any] | None, claim_filter: ClaimFilterObservation
) -> dict[str, Any] | None:
    """Combine stage observations into the single trace field the API exposes.

    A disabled stage contributes nothing, so a query with no optional stages active
    still reports ``None`` exactly as before.
    """
    if claim_filter.status is ClaimFilterStatus.DISABLED:
        return hyde
    merged = dict(hyde or {})
    merged["claim_filter"] = claim_filter.to_dict()
    return merged


def _hyde_trace(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "mode": trace.mode,
        "provider": trace.provider,
        "model": trace.model,
        "latency_ms": trace.latency_ms,
        "status": trace.status,
        "fallback_reason": trace.fallback_reason,
    }
