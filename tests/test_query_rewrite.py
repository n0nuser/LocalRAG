from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest

from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.rag.query_rewrite import expand_query, rewrite_query
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


@dataclass
class ExpansionProvider(FakeProvider):
    answer: str = ""

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        _ = (prompt, context, model)
        return LLMResponse(
            answer=self.answer,
            model="m",
            tokens_used=1,
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )


def test_expand_query_retains_original_and_normalizes_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "localrag.rag.query_rewrite.build_provider",
        lambda _settings: ExpansionProvider(
            answer=(
                '{"queries":[" ERR_QUIC_PROTOCOL_ERROR fix ", " ", 7, '
                '"err_quic_protocol_error fix", "second"]}'
            )
        ),
    )

    result = expand_query(
        "How do I fix ERR_QUIC_PROTOCOL_ERROR?",
        "ERR_QUIC_PROTOCOL_ERROR fix",
        Settings(query_expansion_enabled=True),
    )

    assert result.variants == (
        "How do I fix ERR_QUIC_PROTOCOL_ERROR?",
        "ERR_QUIC_PROTOCOL_ERROR fix",
        "second",
    )
    assert [item.reason for item in result.rejected] == ["empty", "malformed", "duplicate"]


def test_expand_query_provider_failure_falls_back_without_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "localrag.rag.query_rewrite.build_provider", lambda _settings: ExplodingProvider()
    )

    result = expand_query("original", "rewritten", Settings(query_expansion_enabled=True))

    assert result.status == "fallback"
    assert result.variants == ("original", "rewritten")
    assert result.provider_error == "RuntimeError"


def test_expand_query_caps_variants_and_query_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "localrag.rag.query_rewrite.build_provider",
        lambda _settings: ExpansionProvider(
            answer='{"queries":["one", "two", "three", "four", "five", "' + "x" * 501 + '"]}'
        ),
    )

    result = expand_query(
        "original",
        "original",
        Settings(query_expansion_enabled=True, query_expansion_max_variants=3),
    )

    assert result.variants == ("original", "one", "two")
