"""Unit tests for LLM provider abstractions (all external calls are mocked)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from localrag.llm.costs import estimate_cost_usd
from localrag.llm.factory import build_provider
from localrag.llm.providers.anthropic_provider import AnthropicProvider
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.providers.ollama import OllamaProvider
from localrag.llm.providers.openai_provider import OpenAIProvider
from localrag.llm.types import LLMResponse
from localrag.settings import Settings


class _ConcreteProvider(BaseLLMProvider):
    """Minimal concrete implementation to test the ABC contract."""

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        return LLMResponse(
            answer="ok", model="test", tokens_used=1, latency_ms=0.0, estimated_cost_usd=0.0
        )

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        yield {"type": "token", "token": "ok"}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_base_provider_contract() -> None:
    provider = _ConcreteProvider()
    resp = provider.generate("hi", ["context"])
    assert resp.answer == "ok"
    assert list(provider.stream("hi", [])) == [
        {"type": "token", "token": "ok"},
        {"type": "final", "sources": []},
    ]
    assert provider.count_tokens("hello world") == 2


def test_ollama_provider_stream_mocked() -> None:
    lines = [
        '{"message": {"role": "assistant", "content": "Hello"}, "done": false}',
        '{"message": {"role": "assistant", "content": " world"}, "done": true}',
    ]

    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = lines
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_resp
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("localrag.llm.providers.ollama.httpx.Client", return_value=mock_client):
        provider = OllamaProvider(
            base_url="http://localhost:11434",
            default_model="llama3.2",
            system_prompt="Be helpful.",
        )
        events = list(provider.stream("hi", ["some context"]))

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["token"] == "Hello"  # noqa: S105
    assert token_events[1]["token"] == " world"  # noqa: S105
    assert events[-1] == {"type": "final", "sources": []}


def test_ollama_provider_cost_is_zero() -> None:
    assert estimate_cost_usd("_default_ollama", 1000) == 0.0


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("ollama", OllamaProvider),
    ],
)
def test_build_provider_returns_correct_type(
    backend: str, expected_type: type[BaseLLMProvider]
) -> None:
    settings = Settings(
        llm_backend=backend,
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
    )
    provider = build_provider(settings)
    assert isinstance(provider, expected_type)
