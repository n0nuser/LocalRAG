"""OpenAI LLM provider via the ``openai`` SDK."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from localrag.llm.costs import estimate_cost_usd
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o-mini",
        system_prompt: str = (
            "You are a helpful assistant. Answer only based on the provided context."
        ),
    ) -> None:
        self._client = OpenAI(api_key=api_key)
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
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {prompt}"},
        ]
        resp = self._client.chat.completions.create(model=used_model, messages=messages)
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else self.count_tokens(answer)
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
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {prompt}"},
        ]
        with self._client.chat.completions.stream(model=used_model, messages=messages) as stream:
            for text in stream.text_stream:
                yield {"type": "token", "token": text}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())
