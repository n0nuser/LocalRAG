"""Ollama LLM provider — wraps the existing RAGEngine streaming logic."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

import httpx

from localrag.llm.costs import estimate_cost_usd
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.ollama.schemas import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatStreamChunk,
    parse_ollama_json_line,
)
from localrag.rag.prompt import build_prompt


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str,
        default_model: str,
        system_prompt: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._base_url = base_url
        self._default_model = default_model
        self._system_prompt = system_prompt
        self._timeout = timeout_seconds

    def generate(
        self,
        prompt: str,
        context: list[str],
        *,
        model: str | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        chunks: list[str] = []
        for event in self.stream(prompt, context, model=model):
            if event["type"] == "token":
                chunks.append(str(event["token"]))
        latency_ms = (time.perf_counter() - t0) * 1000
        used_model = model or self._default_model
        answer = "".join(chunks).strip()
        tokens = len(answer.split())
        return LLMResponse(
            answer=answer,
            model=used_model,
            tokens_used=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimate_cost_usd("_default_ollama", tokens),
        )

    def stream(
        self,
        prompt: str,
        context: list[str],
        *,
        model: str | None = None,
    ) -> Generator[dict[str, Any]]:
        used_model = model or self._default_model
        full_prompt = build_prompt(
            system_prompt=self._system_prompt,
            question=prompt,
            contexts=[{"text": c} for c in context],
        )
        chat_request = OllamaChatRequest(
            model=used_model,
            messages=[OllamaChatMessage(role="user", content=full_prompt)],
            stream=True,
        )
        with (
            httpx.Client(timeout=self._timeout) as client,
            client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=chat_request.model_dump(mode="json", exclude_none=True),
            ) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = parse_ollama_json_line(line, OllamaChatStreamChunk)
                except ValueError:
                    continue
                msg = chunk.message
                token = msg.content if msg and msg.content else ""
                if token:
                    yield {"type": "token", "token": token}
                if chunk.done:
                    break
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())
