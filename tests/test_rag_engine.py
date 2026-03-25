from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import httpx
import pytest
import respx

from localrag.rag.engine import RAGEngine
from localrag.settings import Settings


@dataclass
class StubRetriever:
    contexts: list[dict[str, object]]

    def retrieve(self, question: str, n_results: int | None = None) -> list[dict[str, object]]:
        _ = (question, n_results)
        return self.contexts


@respx.mock
def test_rag_engine_stream_answer_yields_tokens_and_dedupes_sources() -> None:
    settings = Settings(
        ollama_base_url="http://ollama:11434",
        rag_system_prompt="SYS",
        ollama_llm_model="llm",
    )
    contexts = [
        {"text": "chunk-one", "source": "a.md", "chunk_index": 1},
        # Duplicate (same source + chunk_index) to exercise dedupe.
        {"text": "chunk-one", "source": "a.md", "chunk_index": 1},
    ]
    engine = RAGEngine(settings=settings, retriever=StubRetriever(contexts=contexts))

    url = f"{settings.ollama_base_url}/api/chat"
    ndjson_lines = [
        '{"model":"llm","message":{"role":"assistant","content":"Hello"},"done":false}',
        "not-json",
        "",
        '{"model":"llm","message":{"role":"assistant","content":" world"},"done":false}',
        '{"model":"llm","message":{"role":"assistant","content":""},"done":true}',
    ]
    content = ("\n".join(ndjson_lines) + "\n").encode("utf-8")

    respx.post(url).mock(
        return_value=httpx.Response(200, content=content),
    )

    events = list(engine.stream_answer(question="Q", model="llm", n_results=3))

    token_events = [ev for ev in events if ev["type"] == "token"]
    assert [ev["token"] for ev in token_events] == ["Hello", " world"]

    final = events[-1]
    assert final["type"] == "final"
    assert final["sources"] == [{"source": "a.md", "chunk_index": 1}]


def test_rag_engine_answer_concatenates_tokens_and_returns_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(rag_system_prompt="SYS", ollama_base_url="http://ollama:11434")
    engine = RAGEngine(settings=settings, retriever=StubRetriever(contexts=[]))  # type: ignore[arg-type]

    def fake_stream_answer(
        question: str, model: str | None = None, n_results: int | None = None
    ) -> Generator[dict[str, object]]:
        _ = (question, model, n_results)
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


@respx.mock
def test_rag_engine_stream_answer_raises_on_http_error() -> None:
    settings = Settings(
        ollama_base_url="http://ollama:11434",
        rag_system_prompt="SYS",
        ollama_llm_model="llm",
    )
    engine = RAGEngine(settings=settings, retriever=StubRetriever(contexts=[]))

    url = f"{settings.ollama_base_url}/api/chat"
    respx.post(url).mock(return_value=httpx.Response(500, content=b"oops\n"))

    with pytest.raises(httpx.HTTPStatusError):
        list(engine.stream_answer(question="Q", model="llm", n_results=1))
