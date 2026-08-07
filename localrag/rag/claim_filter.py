"""Bounded scope-applicability filtering of retrieved contexts.

Retrieval selects passages by topical similarity, which is blind to the qualifier
that decides whether a passage actually answers the question. A question about a
single occurrence and a passage about habitual exposure over years are the same
topic, so both are retrieved and both reach the model.

This stage asks the provider one bounded question — which of these passages do not
apply at the scope the question asks about — and drops those before generation. It
never rewrites a passage, never adds one, and never decides the answer: the only
thing it can do is remove a context that was already retrieved.

Failure is always degradation to the unfiltered contexts. Answering with more
context than necessary is the behavior that shipped before this stage existed;
answering with too little because a judgment call went wrong is a regression.

Contract: ADR 041.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from localrag.llm.providers.base import BaseLLMProvider
from localrag.rag.prompt import MAX_SECTION_CHARS
from localrag.settings import Settings

logger = logging.getLogger(__name__)

# The provider is asked for a strict, minimal JSON object rather than prose: the
# smaller the output surface, the less a small local model can drift into
# free-form commentary that no parser can act on safely.
_INSTRUCTION = (
    "You decide which numbered passages do NOT apply to the question's scope.\n"
    "Discard a passage only when its claims are stated at a different scope than "
    "the question asks about - for example the question asks about a single "
    "occurrence and the passage describes repeated, habitual, or long-term "
    "exposure.\n"
    "Do not judge topic, quality, or correctness. When unsure, keep the passage.\n"
    'Reply with only this JSON object: {"inapplicable": [<passage numbers>]}\n'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# A hard ceiling independent of configuration: the filter must never become the
# thing that blows the prompt budget it is meant to protect.
_HARD_MAX_INPUT_CHARS = 4000


class ClaimFilterStatus(StrEnum):
    DISABLED = "disabled"
    NO_CONTEXT = "no_context"
    FILTERED = "filtered"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ClaimFilterObservation:
    """Observable record of what the stage did, surfaced on the query trace."""

    status: ClaimFilterStatus
    evaluated: int = 0
    discarded: int = 0
    latency_ms: float = 0.0
    model: str = ""
    error: str | None = None
    discarded_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": str(self.status),
            "evaluated": self.evaluated,
            "discarded": self.discarded,
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
        }
        if self.error:
            payload["error"] = self.error
        if self.discarded_sources:
            payload["discarded_sources"] = list(self.discarded_sources)
        return payload


@dataclass(frozen=True)
class ClaimFilterResult:
    contexts: list[dict[str, Any]]
    observation: ClaimFilterObservation


def filter_inapplicable_contexts(
    contexts: list[dict[str, Any]],
    question: str,
    settings: Settings,
    provider: BaseLLMProvider,
) -> ClaimFilterResult:
    """Drop retrieved contexts whose scope does not match the question's.

    Returns the contexts unchanged whenever the stage is disabled, has nothing to
    do, or cannot reach a usable verdict.
    """
    model = settings.claim_filter_model or provider.default_model
    if not settings.claim_filter_enabled:
        return ClaimFilterResult(contexts, ClaimFilterObservation(ClaimFilterStatus.DISABLED))
    if not contexts:
        return ClaimFilterResult(contexts, ClaimFilterObservation(ClaimFilterStatus.NO_CONTEXT))

    # Built outside the guard on purpose: a failure here is a bug in this module,
    # not a provider being unreachable, and silently degrading would hide it.
    prompt = _build_prompt(contexts, question, settings)

    started = time.perf_counter()
    try:
        response = provider.generate_from_prompt(prompt, model=model)
        discarded = _parse_indices(response.answer, len(contexts))
    except Exception as exc:
        # Any provider failure degrades to the unfiltered contexts; this stage must
        # never be the reason a valid query fails.
        return _fallback(contexts, started, model, type(exc).__name__)

    if discarded is None:
        return _fallback(contexts, started, model, "unparseable_output")
    if len(discarded) >= len(contexts):
        # Emptying the context set would turn a grounded answer into no answer on
        # the strength of one judgment call. The engine's own abstain path already
        # covers genuinely insufficient evidence.
        return _fallback(contexts, started, model, "all_contexts_discarded")

    kept = [context for index, context in enumerate(contexts, start=1) if index not in discarded]
    latency = (time.perf_counter() - started) * 1000
    sources = tuple(_hit_id(contexts[index - 1]) for index in sorted(discarded))
    if settings.claim_filter_log_content and sources:
        logger.debug("claim_filter_discarded=%s", sources)
    logger.info(
        "claim_filter evaluated=%s discarded=%s latency_ms=%.1f",
        len(contexts),
        len(discarded),
        latency,
    )
    return ClaimFilterResult(
        kept,
        ClaimFilterObservation(
            ClaimFilterStatus.FILTERED,
            evaluated=len(contexts),
            discarded=len(discarded),
            latency_ms=latency,
            model=model,
            discarded_sources=sources,
        ),
    )


def _hit_id(context: dict[str, Any]) -> str:
    """Stable ``source#chunk_index`` identity, matching the retriever's dedupe key."""
    return f"{context.get('source', 'unknown')}#{context.get('chunk_index', -1)}"


def _fallback(
    contexts: list[dict[str, Any]], started: float, model: str, error: str
) -> ClaimFilterResult:
    latency = (time.perf_counter() - started) * 1000
    logger.warning("claim_filter_fallback reason=%s latency_ms=%.1f", error, latency)
    return ClaimFilterResult(
        contexts,
        ClaimFilterObservation(
            ClaimFilterStatus.FALLBACK,
            evaluated=len(contexts),
            latency_ms=latency,
            model=model,
            error=error,
        ),
    )


def _build_prompt(contexts: list[dict[str, Any]], question: str, settings: Settings) -> str:
    max_chars = min(settings.claim_filter_input_max_chars, _HARD_MAX_INPUT_CHARS)
    per_context = max(1, max_chars // max(1, len(contexts)))
    blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        text = str(context.get("expanded_text") or context.get("text") or "")[:per_context]
        metadata = context.get("metadata")
        heading = ""
        if isinstance(metadata, dict):
            heading = str(metadata.get("heading_path") or "").strip()[:MAX_SECTION_CHARS]
        header = f"[{index}]"
        if heading:
            # Scope usually lives in the heading, not the sentence; judging without
            # it is the blind case #172 describes.
            header = f"{header} section={heading}"
        blocks.append(f"{header}\n{text}")
    joined = "\n\n".join(blocks)
    return f"{_INSTRUCTION}\nQuestion:\n{question}\n\nPassages:\n{joined}\n\nJSON:"


def _parse_indices(answer: str, count: int) -> set[int] | None:
    """Return the 1-based indices to discard, or ``None`` when unparseable."""
    match = _JSON_RE.search(answer or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("inapplicable")
    if not isinstance(raw, list):
        return None
    # Out-of-range or non-integer entries are dropped rather than failing the whole
    # verdict: a model that miscounts should cost one passage, not the stage.
    return {value for value in raw if isinstance(value, int) and 1 <= value <= count}
