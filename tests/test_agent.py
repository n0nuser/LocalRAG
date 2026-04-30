"""Tests for the agent service and POST /agent/query endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from localrag.agent.service import AgentResponse, run_agent
from localrag.api.dependencies import get_api_settings, get_engine
from localrag.api.main import app
from localrag.settings import Settings, get_settings


@dataclass
class StubRetriever:
    def retrieve(self, question: str, n_results: int | None = None) -> list[dict[str, Any]]:
        _ = (question, n_results)
        return [{"source": "doc.md", "chunk_index": 0, "text": "chunk"}]


@dataclass
class StubEngine:
    retriever: StubRetriever = field(default_factory=StubRetriever)
    settings: Settings = field(default_factory=lambda: Settings(ollama_llm_model="stub-model"))

    def answer(
        self, question: str, model: str | None = None, n_results: int | None = None
    ) -> dict[str, Any]:
        _ = (model, n_results)
        return {
            "answer": f"Answer to: {question}",
            "sources": [{"source": "doc.md", "chunk_index": 0}],
        }

    def stream_chat_from_contexts(
        self, *, contexts: list[dict[str, Any]], question: str, model: str | None
    ) -> object:
        _ = (contexts, question, model)
        yield {"type": "token", "token": "hello"}
        yield {"type": "final", "sources": []}

    @staticmethod
    def _extract_sources(contexts: list[dict[str, Any]]) -> list[dict[str, object]]:
        return [
            {"source": str(c.get("source", "")), "chunk_index": int(c.get("chunk_index", -1))}
            for c in contexts
        ]


def _make_mock_anthropic_response(tool_name: str, tool_input: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input

    resp = MagicMock()
    resp.content = [block]
    return resp


def test_run_agent_search_documents() -> None:
    mock_resp = _make_mock_anthropic_response("search_documents", {"query": "installation steps"})
    engine = StubEngine()

    with patch("localrag.agent.service.anthropic.Anthropic") as mock_client:
        mock_client.return_value.messages.create.return_value = mock_resp
        result = run_agent(
            question="How do I install?",
            engine=engine,
            api_key="sk-ant-test",
            model="claude-haiku-4-5",
        )

    assert result.tool_used == "search_documents"
    assert "installation steps" in result.reasoning
    assert result.answer != ""


def test_run_agent_answer_directly() -> None:
    mock_resp = _make_mock_anthropic_response(
        "answer_directly",
        {"answer": "Hi! How can I help?", "reasoning": "Greeting, no search needed."},
    )
    engine = StubEngine()

    with patch("localrag.agent.service.anthropic.Anthropic") as mock_client:
        mock_client.return_value.messages.create.return_value = mock_resp
        result = run_agent(
            question="Hello!",
            engine=engine,
            api_key="sk-ant-test",
            model="claude-haiku-4-5",
        )

    assert result.tool_used == "answer_directly"
    assert result.answer == "Hi! How can I help?"
    assert result.sources == []


def test_agent_endpoint_returns_503_without_api_key() -> None:
    app.dependency_overrides[get_engine] = lambda: StubEngine()
    app.dependency_overrides[get_api_settings] = lambda: Settings(anthropic_api_key="")
    app.dependency_overrides[get_settings] = lambda: Settings(anthropic_api_key="")
    client = TestClient(app)

    response = client.post("/agent/query", json={"question": "Hello"})
    assert response.status_code == 503

    app.dependency_overrides.clear()


def test_agent_endpoint_calls_run_agent() -> None:
    app.dependency_overrides[get_engine] = lambda: StubEngine()
    app.dependency_overrides[get_api_settings] = lambda: Settings(anthropic_api_key="sk-ant-test")
    app.dependency_overrides[get_settings] = lambda: Settings(anthropic_api_key="sk-ant-test")
    client = TestClient(app)

    with patch("localrag.api.routers.agent.run_agent") as mock_run:
        mock_run.return_value = AgentResponse(
            answer="42",
            tool_used="answer_directly",
            reasoning="Simple answer.",
            sources=[],
            latency_ms=10.0,
            model="claude-haiku-4-5",
        )
        response = client.post("/agent/query", json={"question": "What is the answer?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "42"
    assert body["tool_used"] == "answer_directly"
    assert "latency_ms" in body

    app.dependency_overrides.clear()
