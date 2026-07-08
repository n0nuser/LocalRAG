"""Retry-with-backoff + circuit-breaker wrapping for any BaseLLMProvider.

Composes two independent concerns: transient failures get retried a few
times with exponential backoff; sustained failure trips the breaker open so
further calls fail fast (or fall back to a secondary provider) instead of
retry-storming a down backend.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from functools import partial
from typing import Any, TypeVar

import anthropic
import httpx
import openai
import pybreaker
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_RETRYABLE_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTION_TYPES):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class ResilientProvider(BaseLLMProvider):
    """Wraps a provider with retry-with-backoff and a circuit breaker."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        max_attempts: int = 3,
        fail_max: int = 5,
        reset_timeout_seconds: float = 30.0,
        fallback_provider: BaseLLMProvider | None = None,
    ) -> None:
        self._provider = provider
        self._fallback_provider = fallback_provider
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=fail_max, reset_timeout=reset_timeout_seconds
        )
        self._max_attempts = max_attempts

    def _retrying(self, func: Callable[[], _T]) -> _T:
        wrapped = retry(
            retry=retry_if_exception(_should_retry),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )(func)
        return wrapped()

    def _call_with_breaker(self, fn: Callable[[], _T], fallback: Callable[[], _T] | None) -> _T:
        try:
            return self._breaker.call(self._retrying, fn)
        except pybreaker.CircuitBreakerError:
            logger.warning("llm_circuit_open falling_back=%s", fallback is not None)
            if fallback is not None:
                return fallback()
            raise

    def _stream_with_breaker(
        self,
        start_fn: Callable[[], tuple[Generator[dict[str, Any]], dict[str, Any] | None]],
        fallback_stream: Callable[[], Generator[dict[str, Any]]] | None,
    ) -> Generator[dict[str, Any]]:
        # Streaming can't be retried mid-token-stream: the retry/breaker call only
        # guards establishing the stream (forcing the first event to surface any
        # connection error), then remaining tokens pass through untouched.
        try:
            gen, first_event = self._breaker.call(self._retrying, start_fn)
        except pybreaker.CircuitBreakerError:
            logger.warning("llm_circuit_open falling_back=%s", fallback_stream is not None)
            if fallback_stream is not None:
                yield from fallback_stream()
                return
            raise
        if first_event is not None:
            yield first_event
        yield from gen

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        fallback = None
        if self._fallback_provider is not None:
            fallback = partial(self._fallback_provider.generate, prompt, context, model=model)
        return self._call_with_breaker(
            partial(self._provider.generate, prompt, context, model=model), fallback
        )

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        def start() -> tuple[Generator[dict[str, Any]], dict[str, Any] | None]:
            gen = self._provider.stream(prompt, context, model=model)
            first_event = next(gen, None)
            return gen, first_event

        fallback_stream = None
        if self._fallback_provider is not None:
            fallback_stream = partial(self._fallback_provider.stream, prompt, context, model=model)
        yield from self._stream_with_breaker(start, fallback_stream)

    def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        fallback = None
        if self._fallback_provider is not None:
            fallback = partial(self._fallback_provider.generate_from_prompt, prompt, model=model)
        return self._call_with_breaker(
            partial(self._provider.generate_from_prompt, prompt, model=model), fallback
        )

    def stream_from_prompt(
        self, prompt: str, *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        def start() -> tuple[Generator[dict[str, Any]], dict[str, Any] | None]:
            gen = self._provider.stream_from_prompt(prompt, model=model)
            first_event = next(gen, None)
            return gen, first_event

        fallback_stream = None
        if self._fallback_provider is not None:
            fallback_stream = partial(
                self._fallback_provider.stream_from_prompt, prompt, model=model
            )
        yield from self._stream_with_breaker(start, fallback_stream)

    def count_tokens(self, text: str) -> int:
        return self._provider.count_tokens(text)
