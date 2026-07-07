"""Build the configured LLM provider from Settings."""

from __future__ import annotations

from localrag.llm.providers.anthropic_provider import AnthropicProvider
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.providers.ollama import OllamaProvider
from localrag.llm.providers.openai_provider import OpenAIProvider
from localrag.llm.resilience import ResilientProvider
from localrag.settings import Settings


def build_provider(settings: Settings) -> BaseLLMProvider:
    """Return the provider selected by ``LLM_BACKEND``, wrapped with retry + circuit breaker."""
    primary = _build_raw_provider(settings, settings.llm_backend)
    fallback = (
        _build_raw_provider(settings, settings.llm_fallback_backend)
        if settings.llm_fallback_backend
        else None
    )
    return ResilientProvider(
        primary,
        max_attempts=settings.llm_retry_max_attempts,
        fail_max=settings.llm_circuit_fail_max,
        reset_timeout_seconds=settings.llm_circuit_reset_timeout_seconds,
        fallback_provider=fallback,
    )


def _build_raw_provider(settings: Settings, backend: str) -> BaseLLMProvider:
    """Return the unwrapped provider for ``backend`` (no retry/circuit-breaker)."""
    backend = backend.lower()
    if backend == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            default_model=settings.openai_model,
            system_prompt=settings.rag_system_prompt,
        )
    if backend == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            default_model=settings.anthropic_model,
            system_prompt=settings.rag_system_prompt,
        )
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_llm_model,
        system_prompt=settings.rag_system_prompt,
    )
