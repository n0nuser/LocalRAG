from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.rag.engine import RAGEngine
from localrag.settings import Settings


@dataclass
class StubRetriever:
    contexts: list[dict[str, object]]

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        _ = (question, n_results, metadata_filter)
        return self.contexts


@dataclass
class FakeProvider(BaseLLMProvider):
    tokens: list[str]
    prompts_seen: list[str] = field(default_factory=list)

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        raise NotImplementedError

    def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def stream_from_prompt(
        self, prompt: str, *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        self.prompts_seen.append(prompt)
        for token in self.tokens:
            yield {"type": "token", "token": token}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_rag_engine_stream_answer_yields_tokens_and_dedupes_sources() -> None:
    settings = Settings(rag_system_prompt="SYS")
    contexts = [
        {"text": "chunk-one", "source": "a.md", "chunk_index": 1},
        # Duplicate (same source + chunk_index) to exercise dedupe.
        {"text": "chunk-one", "source": "a.md", "chunk_index": 1},
    ]
    provider = FakeProvider(tokens=["Hello", " world"])
    engine = RAGEngine(
        settings=settings, retriever=StubRetriever(contexts=contexts), provider=provider
    )

    events = list(engine.stream_answer(question="Q", model="llm", n_results=3))

    token_events = [ev for ev in events if ev["type"] == "token"]
    assert [ev["token"] for ev in token_events] == ["Hello", " world"]

    final = events[-1]
    assert final["type"] == "final"
    assert final["sources"] == [{"source": "a.md", "chunk_index": 1}]
    assert "SYS" in provider.prompts_seen[0]
    assert "chunk-one" in provider.prompts_seen[0]


def test_rag_engine_answer_concatenates_tokens_and_returns_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(rag_system_prompt="SYS")
    provider = FakeProvider(tokens=[])
    engine = RAGEngine(settings=settings, retriever=StubRetriever(contexts=[]), provider=provider)

    def fake_stream_answer(
        question: str,
        model: str | None = None,
        n_results: int | None = None,
        metadata_filter: dict[str, object] | None = None,
    ) -> Generator[dict[str, object]]:
        _ = (question, model, n_results, metadata_filter)
        yield {"type": "token", "token": "Hello "}
        yield {"type": "token", "token": "World"}
        yield {
            "type": "final",
            "sources": [{"source": "a.md", "chunk_index": 1}, {"source": "b.md", "chunk_index": 0}],
        }

    monkeypatch.setattr(engine, "stream_answer", fake_stream_answer)

    out = engine.answer(question="Q", model="m", n_results=3)
    assert out["answer"] == "Hello World"
    assert out["sources"] == [
        {"source": "a.md", "chunk_index": 1},
        {"source": "b.md", "chunk_index": 0},
    ]


def test_rag_engine_stream_answer_propagates_provider_error() -> None:
    settings = Settings(rag_system_prompt="SYS")

    @dataclass
    class ExplodingProvider(BaseLLMProvider):
        def generate(
            self, prompt: str, context: list[str], *, model: str | None = None
        ) -> LLMResponse:
            raise NotImplementedError

        def stream(
            self, prompt: str, context: list[str], *, model: str | None = None
        ) -> Generator[dict[str, Any]]:
            raise NotImplementedError

        def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
            raise NotImplementedError

        def stream_from_prompt(
            self, prompt: str, *, model: str | None = None
        ) -> Generator[dict[str, Any]]:
            raise RuntimeError("provider down")
            yield  # pragma: no cover — unreachable, keeps this a generator

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    engine = RAGEngine(
        settings=settings, retriever=StubRetriever(contexts=[]), provider=ExplodingProvider()
    )

    with pytest.raises(RuntimeError, match="provider down"):
        list(engine.stream_answer(question="Q", model="llm", n_results=1))
