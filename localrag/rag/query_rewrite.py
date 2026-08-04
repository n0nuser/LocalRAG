"""Bounded LLM query transformation for retrieval only.

Rewriting and expansion are separate operations.  When both are enabled the
order is exactly: original question -> rewrite -> expansion -> retrieval.
Expansion makes one provider call and always includes the original question;
the final answer prompt is never built from generated queries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from localrag.llm.factory import build_provider
from localrag.settings import Settings

logger = logging.getLogger(__name__)

_REWRITE_INSTRUCTION = (
    "Rewrite the user's question as a short, keyword-dense search query for a "
    "document retrieval system. Keep any exact identifiers, codes, or names "
    "verbatim. Respond with only the rewritten query, no explanation."
)

_EXPANSION_INSTRUCTION = (
    "Generate alternative search queries for the user's retrieval query. Return only a JSON "
    'object like {"queries":["..."]}; no explanations or facts. Preserve every exact '
    "identifier, code, version, and proper name verbatim."
)
_HARD_MAX_VARIANTS = 8
_HARD_MAX_QUERY_CHARS = 500


@dataclass(frozen=True)
class RejectedVariant:
    value: object
    reason: Literal["empty", "duplicate", "malformed", "too_long"]


@dataclass(frozen=True)
class QueryExpansionResult:
    original: str
    rewrite: str | None
    variants: tuple[str, ...]
    rejected: tuple[RejectedVariant, ...]
    status: Literal["disabled", "expanded", "fallback"]
    provider_error: str | None = None


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _parse_expansion_output(raw: str) -> list[object]:
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("queries")
    if not isinstance(parsed, list):
        raise TypeError("expansion response must contain a JSON list")
    return parsed


def expand_query(
    original: str,
    search_query: str,
    settings: Settings,
    *,
    rewrite: str | None = None,
) -> QueryExpansionResult:
    """Return bounded retrieval variants, retaining ``original`` exactly.

    Provider failures and malformed responses are deliberately non-fatal.  The
    fallback retains the original and rewritten query (when distinct), so
    enabling expansion cannot make retrieval fail or leak model commentary into search.
    """
    fallback = (
        (original,)
        if _normalized(original) == _normalized(search_query)
        else (original, search_query)
    )
    if not settings.query_expansion_enabled:
        return QueryExpansionResult(original, rewrite, (search_query,), (), "disabled")

    try:
        provider = build_provider(
            settings.model_copy(update={"rag_system_prompt": _EXPANSION_INSTRUCTION})
        )
        response = provider.generate(search_query, context=[])
        raw_variants = _parse_expansion_output(response.answer.strip())
    except Exception as exc:
        logger.exception("query_expansion_failed_falling_back")
        return QueryExpansionResult(original, rewrite, fallback, (), "fallback", type(exc).__name__)

    variants = [original]
    seen = {_normalized(original)}
    rejected: list[RejectedVariant] = []
    for value in raw_variants:
        if len(variants) >= min(settings.query_expansion_max_variants, _HARD_MAX_VARIANTS):
            break
        if not isinstance(value, str):
            rejected.append(RejectedVariant(value, "malformed"))
            continue
        candidate = value.strip()
        if not candidate:
            rejected.append(RejectedVariant(value, "empty"))
        elif len(candidate) > min(settings.query_expansion_max_query_chars, _HARD_MAX_QUERY_CHARS):
            rejected.append(RejectedVariant(value, "too_long"))
        elif _normalized(candidate) in seen:
            rejected.append(RejectedVariant(value, "duplicate"))
        else:
            variants.append(candidate)
            seen.add(_normalized(candidate))
    return QueryExpansionResult(original, rewrite, tuple(variants), tuple(rejected), "expanded")


def rewrite_query(question: str, settings: Settings) -> str:
    """Return a keyword-dense reformulation of ``question`` for retrieval only.

    Falls back to the original ``question`` on any provider failure. The
    original question — not this rewrite — is still used for the final
    answer prompt; rewriting is a retrieval-only concern.
    """
    rewrite_provider = build_provider(
        settings.model_copy(update={"rag_system_prompt": _REWRITE_INSTRUCTION})
    )
    try:
        response = rewrite_provider.generate(question, context=[])
    except Exception:
        logger.exception("query_rewrite_failed_falling_back_to_original")
        return question
    rewritten = response.answer.strip()
    return rewritten or question
