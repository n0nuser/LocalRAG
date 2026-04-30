"""Build the configured LLM provider from Settings."""

from __future__ import annotations

from localrag.llm.providers.anthropic_provider import AnthropicProvider
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.providers.ollama import OllamaProvider
from localrag.llm.providers.openai_provider import OpenAIProvider
from localrag.settings import Settings


def build_provider(settings: Settings) -> BaseLLMProvider:
    """Return the provider selected by ``LLM_BACKEND``."""
    backend = settings.llm_backend.lower()
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
