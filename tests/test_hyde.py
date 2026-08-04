from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import pytest
from typer.testing import CliRunner

from localrag.cli.app import app
from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.rag.hyde import generate_hypothetical
from localrag.rag.retriever import Retriever
from localrag.settings import Settings


@dataclass
class FakeProvider(BaseLLMProvider):
    answer: str = "A useful hypothetical passage."
    prompts: list[str] = field(default_factory=list)
    model_name: str = "local-model"

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(self.answer, model or self.model_name, 3, 4.0, 0.0)

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
        return len(text.split())


def test_hyde_disabled_does_not_build_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localrag.rag.hyde.build_provider", lambda _: pytest.fail("disabled"))
    text, observation = generate_hypothetical("question", Settings())
    assert text is None
    assert observation.status == "disabled"


def test_hyde_prompt_and_output_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider()
    monkeypatch.setattr("localrag.rag.hyde.build_provider", lambda _: provider)
    text, observation = generate_hypothetical(
        "x" * 100,
        Settings(hyde_enabled=True, hyde_input_max_chars=10, hyde_output_max_chars=100),
    )
    assert text == "A useful hypothetical passage."
    assert len(provider.prompts[-1].split("User question: ", 1)[1]) == 10
    assert observation.status == "generated"
    assert observation.hypothetical is None


@pytest.mark.parametrize("answer", ["", '{"answer": "hidden"}', "```text\nnope\n```"])
def test_hyde_invalid_output_falls_back(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    monkeypatch.setattr("localrag.rag.hyde.build_provider", lambda _: FakeProvider(answer=answer))
    text, observation = generate_hypothetical("question", Settings(hyde_enabled=True))
    assert text is None
    assert observation.status == "fallback"


@pytest.mark.parametrize("error", [RuntimeError("provider"), TimeoutError("timeout")])
def test_hyde_provider_error_and_timeout_fall_back(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def build(_: Settings) -> BaseLLMProvider:
        raise error

    monkeypatch.setattr("localrag.rag.hyde.build_provider", build)
    text, observation = generate_hypothetical("question", Settings(hyde_enabled=True))
    assert text is None
    assert observation.status == "fallback"
    assert observation.fallback_reason == type(error).__name__


def test_hyde_rejects_non_local_provider_without_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localrag.rag.hyde.build_provider", lambda _: pytest.fail("remote call"))
    _, observation = generate_hypothetical(
        "question", Settings(hyde_enabled=True, llm_backend="openai")
    )
    assert observation.status == "unsupported_provider"


@dataclass
class Embedder:
    inputs: list[str] = field(default_factory=list)

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [1.0, 0.0]


@dataclass
class Store:
    def query(
        self, embedding: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


@dataclass
class Bm25:
    inputs: list[str] = field(default_factory=list)

    def query(self, text: str, top_k: int) -> list[Any]:
        self.inputs.append(text)
        return []


def test_hyde_uses_hypothetical_dense_and_original_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider(answer="hypothetical answer")
    monkeypatch.setattr("localrag.rag.hyde.build_provider", lambda _: provider)
    embedder, bm25 = Embedder(), Bm25()
    retriever = Retriever(
        Settings(hyde_enabled=True, retrieval_mode="hybrid"),
        embedder,
        Store(),
        bm25_index=bm25,  # type: ignore[arg-type]
    )
    retriever.retrieve("original question")
    assert embedder.inputs == ["hypothetical answer"]
    assert bm25.inputs == ["original question"]
    assert retriever.last_hyde is not None
    assert retriever.last_hyde.hypothetical is None


def test_experiment_mode_controls_rewrite_hyde_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def rewrite(_question: str, _settings: Settings) -> str:
        calls.append("rewrite")
        return "rewritten"

    monkeypatch.setattr("localrag.rag.retriever.rewrite_query", rewrite)
    monkeypatch.setattr("localrag.rag.hyde.build_provider", lambda _: FakeProvider(answer="hypo"))
    retriever = Retriever(
        Settings(retrieval_experiment_mode="rewrite+hyde", retrieval_mode="vector"),
        Embedder(),
        Store(),
    )  # type: ignore[arg-type]
    retriever.retrieve("original")
    assert calls == ["rewrite"]
    assert retriever.last_hyde is not None
    assert retriever.last_hyde.mode == "hyde"


def test_hyde_settings_snapshot_is_stable_and_redacted() -> None:
    settings = Settings(hyde_enabled=True, hyde_model="local-model")
    assert settings.resolved_snapshot()["hyde_enabled"] is True
    assert settings.resolved_snapshot()["hyde_model"] == "local-model"


def test_cli_query_surfaces_bounded_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    class Engine:
        def stream_answer(self, **kwargs: object) -> Generator[dict[str, object]]:
            yield {"type": "token", "token": "answer"}
            yield {
                "type": "final",
                "sources": [],
                "trace": {"mode": "hyde", "status": "generated"},
            }

    monkeypatch.setattr("localrag.cli.commands.query.get_engine", lambda: Engine())
    result = CliRunner().invoke(app, ["query", "question"])
    assert result.exit_code == 0
    assert "answer" in result.stdout
    assert 'trace={"mode": "hyde", "status": "generated"}' in result.stdout
