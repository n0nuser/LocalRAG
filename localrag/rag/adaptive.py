"""Bounded evidence-policy controller for adaptive retrieval.

This module deliberately contains no answer-generation reasoning.  Provider
outputs are accepted only as structured retrieval refinements or critiques and
the original question is never replaced for final generation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from localrag.llm.providers.base import BaseLLMProvider
from localrag.rag.retriever import Retriever
from localrag.settings import Settings


class RetrievalState(StrEnum):
    INITIAL_RETRIEVE = "initial_retrieve"
    EVALUATE_EVIDENCE = "evaluate_evidence"
    ESCALATE = "escalate"
    REFINE = "refine"
    RETRIEVE = "retrieve"
    ANSWER = "answer"
    ABSTAIN = "abstain"
    DONE = "done"


class QueryKind(StrEnum):
    ORIGINAL = "original"
    REFINED = "refined"


class StopReason(StrEnum):
    NONE = "none"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EMPTY_CORPUS = "empty_corpus"
    FILTERED_NO_RESULTS = "metadata_filtered_no_results"
    REPEATED_EVIDENCE = "repeated_evidence"
    INVALID_REFINEMENT = "invalid_refinement"
    PROVIDER_FAILURE = "provider_failure"
    TIMEOUT = "timeout"
    BUDGET = "budget_exhausted"
    CONTEXT_OVERFLOW = "context_overflow"


@dataclass(frozen=True)
class EvidenceSignals:
    non_empty: bool
    top_score: float
    score_margin: float
    source_diversity: int
    query_coverage: float
    critique_supported: int = 0
    critique_missing: int = 0


@dataclass(frozen=True)
class TraceEvent:
    state: RetrievalState
    round: int
    query_kind: QueryKind
    requested_k: int
    returned_k: int
    hit_ids: tuple[str, ...] = ()
    evidence: EvidenceSignals | None = None
    decision: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: StopReason = StopReason.NONE


@dataclass
class AdaptiveTrace:
    policy: str = "adaptive-retrieval/v1"
    original_query: str = ""
    transitions: list[TraceEvent] = field(default_factory=list)
    stop_reason: StopReason = StopReason.NONE
    abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["transitions"] = [asdict(item) for item in self.transitions]
        return result


@dataclass(frozen=True)
class AdaptiveResult:
    contexts: list[dict[str, Any]]
    trace: AdaptiveTrace


def _hit_id(hit: dict[str, Any]) -> str:
    return f"{hit.get('source', 'unknown')}#{int(hit.get('chunk_index', -1))}"


def _signals(question: str, contexts: list[dict[str, Any]]) -> EvidenceSignals:
    scores = sorted((float(hit.get("score", 0.0)) for hit in contexts), reverse=True)
    terms = {term.casefold() for term in question.split() if len(term) > 2}
    text = " ".join(str(hit.get("text", "")) for hit in contexts).casefold()
    coverage = len([term for term in terms if term in text]) / len(terms) if terms else 1.0
    return EvidenceSignals(
        non_empty=bool(contexts),
        top_score=scores[0] if scores else 0.0,
        score_margin=(scores[0] - scores[1]) if len(scores) > 1 else scores[0] if scores else 0.0,
        source_diversity=len({str(hit.get("source", "unknown")) for hit in contexts}),
        query_coverage=coverage,
    )


def _parse_refinement(raw: str, max_chars: int) -> str | None:
    try:
        value = json.loads(raw).get("query")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value if value and len(value) <= max_chars else None


def _parse_critique(raw: str) -> tuple[int, int] | None:
    try:
        value = json.loads(raw)
        supported = value.get("supported_claims")
        missing = value.get("missing_evidence")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(supported, list) or not isinstance(missing, list):
        return None
    if not all(isinstance(item, str) for item in [*supported, *missing]):
        return None
    return len(supported), len(missing)


class AdaptiveRetrievalPolicy:
    """Run one bounded retrieval policy and return evidence plus an observable trace."""

    def __init__(self, settings: Settings, retriever: Retriever, provider: BaseLLMProvider):
        self.settings = settings
        self.retriever = retriever
        self.provider = provider

    def run(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        question: str,
        *,
        model: str | None = None,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AdaptiveResult:
        started = time.monotonic()
        trace = AdaptiveTrace(original_query=question)
        contexts: list[dict[str, Any]] = []
        seen: set[str] = set()
        query = question
        kind = QueryKind.ORIGINAL
        requested = n_results or self.settings.adaptive_initial_top_k
        rounds = 0
        refinements = 0
        provider_calls = 0
        while rounds < self.settings.adaptive_max_rounds:
            if (time.monotonic() - started) * 1000 > self.settings.adaptive_max_latency_ms:
                return self._stop(trace, contexts, StopReason.TIMEOUT)
            state = RetrievalState.INITIAL_RETRIEVE if rounds == 0 else RetrievalState.RETRIEVE
            rounds += 1
            retrieved = self.retriever.retrieve(
                query, n_results=requested, metadata_filter=metadata_filter
            )
            new_hits = [hit for hit in retrieved if _hit_id(hit) not in seen]
            seen.update(_hit_id(hit) for hit in retrieved)
            contexts.extend(new_hits)
            signals = _signals(question, contexts)
            if self.settings.adaptive_critique_enabled:
                if provider_calls >= self.settings.adaptive_max_provider_calls:
                    return self._stop(trace, contexts, StopReason.BUDGET)
                provider_calls += 1
                try:
                    critique = self.provider.generate(
                        '{"supported_claims": [], "missing_evidence": []} '
                        "Return only this JSON shape. Evaluate evidence against the original "
                        "question: " + question,
                        context=[str(hit.get("text", "")) for hit in contexts],
                        model=model,
                    )
                    parsed_critique = _parse_critique(critique.answer)
                except Exception:
                    return self._stop(trace, contexts, StopReason.PROVIDER_FAILURE)
                if parsed_critique is None:
                    return self._stop(trace, contexts, StopReason.PROVIDER_FAILURE)
                signals = replace(
                    signals,
                    critique_supported=parsed_critique[0],
                    critique_missing=parsed_critique[1],
                )
            context_tokens = sum(len(str(hit.get("text", "")).split()) for hit in contexts)
            if context_tokens > self.settings.llm_context_window_tokens:
                return self._stop(trace, contexts, StopReason.CONTEXT_OVERFLOW)
            trace.transitions.append(
                TraceEvent(
                    state=state,
                    round=rounds,
                    query_kind=kind,
                    requested_k=requested,
                    returned_k=len(retrieved),
                    hit_ids=tuple(_hit_id(hit) for hit in retrieved),
                    evidence=signals,
                    decision="evaluate",
                )
            )
            if not retrieved and rounds == 1:
                return self._stop(
                    trace,
                    contexts,
                    StopReason.FILTERED_NO_RESULTS if metadata_filter else StopReason.EMPTY_CORPUS,
                )
            trace.transitions.append(
                TraceEvent(
                    state=RetrievalState.EVALUATE_EVIDENCE,
                    round=rounds,
                    query_kind=kind,
                    requested_k=requested,
                    returned_k=len(retrieved),
                    hit_ids=tuple(_hit_id(hit) for hit in retrieved),
                    evidence=signals,
                    decision="answer" if self._sufficient(signals) else "escalate",
                )
            )
            if self._sufficient(signals):
                trace.transitions.append(
                    TraceEvent(
                        RetrievalState.ANSWER,
                        rounds,
                        kind,
                        requested,
                        len(contexts),
                        tuple(seen),
                        decision="answer",
                    )
                )
                trace.transitions.append(
                    TraceEvent(
                        RetrievalState.DONE, rounds, kind, requested, len(contexts), tuple(seen)
                    )
                )
                return AdaptiveResult(contexts, trace)
            if rounds >= self.settings.adaptive_max_rounds:
                return self._stop(trace, contexts, StopReason.INSUFFICIENT_EVIDENCE)
            if len(seen) == len({_hit_id(hit) for hit in contexts}) and not new_hits:
                return self._stop(trace, contexts, StopReason.REPEATED_EVIDENCE)
            if requested < self.settings.adaptive_escalation_top_k:
                requested = self.settings.adaptive_escalation_top_k
                trace.transitions.append(
                    TraceEvent(
                        RetrievalState.ESCALATE,
                        rounds,
                        kind,
                        requested,
                        len(contexts),
                        tuple(seen),
                        decision="escalate",
                    )
                )
                continue
            if refinements >= self.settings.adaptive_max_refinements:
                return self._stop(trace, contexts, StopReason.BUDGET)
            refinements += 1
            trace.transitions.append(
                TraceEvent(
                    RetrievalState.REFINE,
                    rounds,
                    kind,
                    requested,
                    len(contexts),
                    tuple(seen),
                    decision="refine",
                )
            )
            try:
                prompt = (
                    '{"query": "..."} Return one JSON object with a retrieval-only query '
                    "that addresses the evidence gap. Original question: " + question
                )
                response = self.provider.generate(
                    prompt,
                    context=[],
                    model=model,
                )
                refined = _parse_refinement(
                    response.answer, self.settings.adaptive_refinement_max_chars
                )
            except Exception:
                return self._stop(trace, contexts, StopReason.PROVIDER_FAILURE)
            if refined is None or refined.casefold() == query.casefold():
                return self._stop(trace, contexts, StopReason.INVALID_REFINEMENT)
            query, kind = refined, QueryKind.REFINED
        return self._stop(trace, contexts, StopReason.BUDGET)

    def _sufficient(self, signals: EvidenceSignals) -> bool:
        return (
            signals.non_empty
            and signals.top_score >= self.settings.adaptive_min_top_score
            and signals.score_margin >= self.settings.adaptive_min_score_margin
            and signals.source_diversity >= self.settings.adaptive_min_source_diversity
            and signals.query_coverage >= self.settings.adaptive_min_query_coverage
        )

    @staticmethod
    def _stop(
        trace: AdaptiveTrace, contexts: list[dict[str, Any]], reason: StopReason
    ) -> AdaptiveResult:
        trace.stop_reason = reason
        trace.abstained = True
        trace.transitions.append(
            TraceEvent(
                RetrievalState.ABSTAIN,
                len(trace.transitions),
                QueryKind.ORIGINAL,
                0,
                len(contexts),
                tuple(_hit_id(hit) for hit in contexts),
                decision="abstain",
                stop_reason=reason,
            )
        )
        trace.transitions.append(
            TraceEvent(
                RetrievalState.DONE,
                len(trace.transitions),
                QueryKind.ORIGINAL,
                0,
                len(contexts),
                tuple(_hit_id(hit) for hit in contexts),
                stop_reason=reason,
            )
        )
        return AdaptiveResult(contexts, trace)
