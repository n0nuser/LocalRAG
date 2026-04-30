"""Anthropic LLM provider via the ``anthropic`` SDK."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

import anthropic

from localrag.llm.costs import estimate_cost_usd
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse

_MAX_TOKENS = 4096


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-haiku-4-5",
        system_prompt: str = (
            "You are a helpful assistant. Answer only based on the provided context."
        ),
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._default_model = default_model
        self._system_prompt = system_prompt

    def generate(
        self,
        prompt: str,
        context: list[str],
        *,
        model: str | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        used_model = model or self._default_model
        context_block = "\n\n".join(context)
        user_content = f"Context:\n{context_block}\n\nQuestion: {prompt}"
        resp = self._client.messages.create(
            model=used_model,
            max_tokens=_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = next((b.text for b in resp.content if hasattr(b, "text")), "")  # type: ignore[union-attr]
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        return LLMResponse(
            answer=answer.strip(),
            model=used_model,
            tokens_used=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimate_cost_usd(used_model, tokens),
        )

    def stream(
        self,
        prompt: str,
        context: list[str],
        *,
        model: str | None = None,
    ) -> Generator[dict[str, Any]]:
        used_model = model or self._default_model
        context_block = "\n\n".join(context)
        user_content = f"Context:\n{context_block}\n\nQuestion: {prompt}"
        with self._client.messages.stream(
            model=used_model,
            max_tokens=_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "token", "token": text}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())
