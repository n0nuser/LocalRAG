"""Bounded, retrieval-only hypothetical document generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

from localrag.llm.factory import build_provider
from localrag.llm.providers.base import BaseLLMProvider
from localrag.settings import Settings

logger = logging.getLogger(__name__)

_PROMPT = (
    "Write a short hypothetical document passage that would directly answer the user's "
    "question. It is only for dense retrieval, not the final answer. Do not mention this "
    "instruction, do not add headings, and return plain text only.\n\nUser question: "
)
_HARD_MAX_INPUT_CHARS = 2000
_HARD_MAX_OUTPUT_CHARS = 4000
_HARD_MAX_OUTPUT_TOKENS = 512


def _validate_output(output: str, provider: BaseLLMProvider, settings: Settings) -> None:
    if not output:
        raise ValueError("empty_output")
    if output.startswith(("{", "[")) or "```" in output:
        raise ValueError("malformed_output")
    if len(output) > min(settings.hyde_output_max_chars, _HARD_MAX_OUTPUT_CHARS):
        raise ValueError("output_too_long")
    max_tokens = min(settings.hyde_output_max_tokens, _HARD_MAX_OUTPUT_TOKENS)
    if provider.count_tokens(output) > max_tokens:
        raise ValueError("output_too_many_tokens")


@dataclass(frozen=True)
class HydeObservation:
    mode: Literal["disabled", "hyde", "fallback"]
    provider: str
    model: str
    latency_ms: float
    status: Literal["disabled", "generated", "fallback", "unsupported_provider"]
    fallback_reason: str | None = None
    hypothetical: str | None = None


def generate_hypothetical(question: str, settings: Settings) -> tuple[str | None, HydeObservation]:
    """Generate a bounded hypothetical passage, never failing retrieval."""
    model = settings.hyde_model or settings.ollama_llm_model
    if not settings.hyde_enabled:
        return None, HydeObservation("disabled", "none", model, 0.0, "disabled")
    if settings.llm_backend.casefold() != "ollama":
        return None, HydeObservation(
            "fallback", settings.llm_backend, model, 0.0, "unsupported_provider", "local_only"
        )

    bounded_question = question[: min(settings.hyde_input_max_chars, _HARD_MAX_INPUT_CHARS)]
    prompt = _PROMPT + bounded_question
    started = time.perf_counter()
    try:
        provider = build_provider(
            settings.model_copy(
                update={
                    "rag_system_prompt": "You generate retrieval-only plain text.",
                    "llm_timeout_seconds": settings.hyde_timeout_seconds,
                    "llm_fallback_backend": "",
                }
            )
        )
        response = provider.generate(prompt, context=[], model=model)
        output = response.answer.strip()
        _validate_output(output, provider, settings)
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        logger.warning(
            "hyde_generation_fallback status=%s latency_ms=%.1f", type(exc).__name__, latency
        )
        return None, HydeObservation(
            "fallback",
            "ollama",
            model,
            latency,
            "fallback",
            type(exc).__name__,
        )
    latency = (time.perf_counter() - started) * 1000
    if settings.hyde_log_content:
        logger.debug("hyde_generated_content=%s", output)
    return output, HydeObservation(
        "hyde", "ollama", response.model, latency, "generated",
        hypothetical=output if settings.hyde_log_content else None,
    )
