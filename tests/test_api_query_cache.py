from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from localrag.api.dependencies import get_engine, get_query_cache
from localrag.api.main import app
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.rag.engine import RAGEngine
from localrag.rag.query_cache import QueryCache
from localrag.settings import Settings


@dataclass
class CountingRetriever:
    calls: list[str]
    contexts: list[dict[str, object]]

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append(question)
        return self.contexts


@dataclass
class FakeProvider(BaseLLMProvider):
    """Minimal stub standing in for the real LLM provider (see tests/test_rag_engine.py)."""

    tokens: list[str] = field(default_factory=lambda: ["answer"])

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
        for token in self.tokens:
            yield {"type": "token", "token": token}
        yield {"type": "final", "sources": []}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_repeat_query_is_served_from_cache_without_calling_retriever() -> None:
    settings = Settings(ollama_base_url="http://ollama:11434", ollama_llm_model="llm")
    retriever = CountingRetriever(calls=[], contexts=[])
    engine = RAGEngine(settings=settings, retriever=retriever, provider=FakeProvider())  # type: ignore[arg-type]
    cache = QueryCache(maxsize=10, ttl_seconds=60)

    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_query_cache] = lambda: cache
    client = TestClient(app)

    first = client.post("/query", json={"question": "What is X?"})
    second = client.post("/query", json={"question": "What is X?"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert retriever.calls == ["What is X?"]  # second request served from cache

    app.dependency_overrides.clear()
