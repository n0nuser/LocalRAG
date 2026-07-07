from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pybreaker
import pytest

from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.resilience import ResilientProvider
from localrag.llm.types import LLMResponse


@dataclass
class FlakyProvider(BaseLLMProvider):
    """Fails ``fail_times`` times with a retryable error, then succeeds."""

    fail_times: int
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        self.calls.append(prompt)
        if len(self.calls) <= self.fail_times:
            request = httpx.Request("POST", "http://x/api/chat")
            raise httpx.ConnectError("boom", request=request)
        return LLMResponse(
            answer="ok", model="m", tokens_used=1, latency_ms=0.0, estimated_cost_usd=0.0
        )

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        yield {"type": "token", "token": "ok"}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@dataclass
class AlwaysFailsProvider(BaseLLMProvider):
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        self.calls.append(prompt)
        request = httpx.Request("POST", "http://x/api/chat")
        raise httpx.ConnectError("always down", request=request)

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@dataclass
class FallbackProvider(BaseLLMProvider):
    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        return LLMResponse(
            answer="fallback", model="fb", tokens_used=1, latency_ms=0.0, estimated_cost_usd=0.0
        )

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        yield {"type": "token", "token": "fallback"}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_resilient_provider_retries_transient_failure_then_succeeds() -> None:
    flaky = FlakyProvider(fail_times=2)
    resilient = ResilientProvider(flaky, max_attempts=3, fail_max=5, reset_timeout_seconds=30)

    response = resilient.generate("q", [])

    assert response.answer == "ok"
    assert len(flaky.calls) == 3


def test_resilient_provider_opens_circuit_after_fail_max_and_falls_back() -> None:
    # pybreaker (>=1) raises CircuitBreakerError directly from the call that
    # crosses fail_max, rather than propagating the underlying exception on
    # that final call — so with fail_max=2, the first call surfaces the
    # provider's own error; the second is where the breaker trips, and since a
    # fallback is configured, ResilientProvider catches that and serves the
    # fallback answer immediately instead of raising.
    always_fails = AlwaysFailsProvider()
    fallback = FallbackProvider()
    resilient = ResilientProvider(
        always_fails,
        max_attempts=1,
        fail_max=2,
        reset_timeout_seconds=30,
        fallback_provider=fallback,
    )

    with pytest.raises(httpx.ConnectError):
        resilient.generate("q", [])

    # Circuit is now open — further calls go straight to the fallback, no further
    # attempts against the primary provider.
    response = resilient.generate("q", [])
    assert response.answer == "fallback"
    assert len(always_fails.calls) == 2

    response_again = resilient.generate("q", [])
    assert response_again.answer == "fallback"
    assert len(always_fails.calls) == 2


def test_resilient_provider_raises_when_circuit_open_and_no_fallback() -> None:
    # With fail_max=1, the very first failure already crosses the threshold, so
    # pybreaker raises CircuitBreakerError on that first call too (see note above).
    always_fails = AlwaysFailsProvider()
    resilient = ResilientProvider(
        always_fails, max_attempts=1, fail_max=1, reset_timeout_seconds=30
    )

    with pytest.raises(pybreaker.CircuitBreakerError):
        resilient.generate("q", [])

    with pytest.raises(pybreaker.CircuitBreakerError):
        resilient.generate("q", [])
