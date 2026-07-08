"""Optional LLM-based query rewriting for retrieval (not for the final answer prompt)."""

from __future__ import annotations

import logging

from localrag.llm.factory import build_provider
from localrag.settings import Settings

logger = logging.getLogger(__name__)

_REWRITE_INSTRUCTION = (
    "Rewrite the user's question as a short, keyword-dense search query for a "
    "document retrieval system. Keep any exact identifiers, codes, or names "
    "verbatim. Respond with only the rewritten query, no explanation."
)


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
