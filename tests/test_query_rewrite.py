from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest

from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.rag.query_rewrite import rewrite_query
from localrag.settings import Settings


@dataclass
class FakeProvider(BaseLLMProvider):
    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        assert prompt == "how do i fix ERR_QUIC_PROTOCOL_ERROR"
        return LLMResponse(
            answer="ERR_QUIC_PROTOCOL_ERROR fix",
            model="m",
            tokens_used=3,
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        raise NotImplementedError

    def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def stream_from_prompt(
        self, prompt: str, *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


@dataclass
class ExplodingProvider(BaseLLMProvider):
    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        raise RuntimeError("boom")

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        raise NotImplementedError

    def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def stream_from_prompt(
        self, prompt: str, *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


def test_rewrite_query_returns_provider_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "localrag.rag.query_rewrite.build_provider", lambda _settings: FakeProvider()
    )

    out = rewrite_query("how do i fix ERR_QUIC_PROTOCOL_ERROR", Settings())

    assert out == "ERR_QUIC_PROTOCOL_ERROR fix"


def test_rewrite_query_falls_back_to_original_on_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "localrag.rag.query_rewrite.build_provider", lambda _settings: ExplodingProvider()
    )

    out = rewrite_query("original question", Settings())

    assert out == "original question"
