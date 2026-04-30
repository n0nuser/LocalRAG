"""Abstract base class for LLM provider implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from localrag.llm.types import LLMResponse


class BaseLLMProvider(ABC):
    """Contract every LLM backend must fulfil."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[str],
        *,
        model: str | None = None,
    ) -> LLMResponse:
        """Return a complete response (blocks until done)."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        context: list[str],
        *,
        model: str | None = None,
    ) -> Generator[dict[str, Any]]:
        """Yield token/final events matching the existing RAG engine contract."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Approximate token count for ``text``."""
