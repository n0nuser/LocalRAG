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
    assert final["sources"] == [
        {"source": "a.md", "chunk_index": 1, "heading_path": None, "chunk_type": None}
    ]
    assert "SYS" in provider.prompts_seen[0]
    assert "chunk-one" in provider.prompts_seen[0]


def test_rag_engine_extract_sources_includes_heading_path_and_chunk_type() -> None:
    settings = Settings(rag_system_prompt="SYS")
    contexts = [
        {
            "text": "chunk",
            "source": "guide.md",
            "chunk_index": 2,
            "metadata": {"heading_path": "Setup > Install", "chunk_type": "markdown_section"},
        }
    ]
    engine = RAGEngine(
        settings=settings,
        retriever=StubRetriever(contexts=contexts),
        provider=FakeProvider(tokens=[]),
    )

    sources = engine._extract_sources(contexts)  # noqa: SLF001

    assert sources == [
        {
            "source": "guide.md",
            "chunk_index": 2,
            "heading_path": "Setup > Install",
            "chunk_type": "markdown_section",
        }
    ]


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


def test_rag_engine_short_circuits_on_low_confidence_context() -> None:
    settings = Settings(rag_min_context_score=0.5)
    contexts = [{"text": "weak match", "source": "a.md", "chunk_index": 0, "score": 0.1}]
    provider = FakeProvider(tokens=["should not be used"])
    engine = RAGEngine(
        settings=settings, retriever=StubRetriever(contexts=contexts), provider=provider
    )

    events = list(engine.stream_answer(question="Q", n_results=1))

    token_events = [ev for ev in events if ev["type"] == "token"]
    expected_refusal = "I don't have enough information in the ingested documents to answer that."
    assert token_events[0]["token"] == expected_refusal
    final = events[-1]
    assert final["low_confidence"] is True
    assert final["sources"] == []
    assert provider.prompts_seen == []


def test_rag_engine_empty_contexts_are_low_confidence_when_threshold_enabled() -> None:
    settings = Settings(rag_min_context_score=0.1)
    provider = FakeProvider(tokens=[])
    engine = RAGEngine(settings=settings, retriever=StubRetriever(contexts=[]), provider=provider)

    events = list(engine.stream_answer(question="Q", n_results=1))
    final = events[-1]
    assert final["low_confidence"] is True


def test_rag_engine_low_confidence_disabled_by_default() -> None:
    settings = Settings()
    assert settings.rag_min_context_score == 0.0
