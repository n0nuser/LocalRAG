from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest

from localrag.llm.providers.base import BaseLLMProvider
from localrag.llm.types import LLMResponse
from localrag.rag.adaptive import AdaptiveRetrievalPolicy, StopReason
from localrag.settings import Settings


@dataclass
class StubRetriever:
    results: list[list[dict[str, Any]]]
    questions: list[str] | None = None

    def retrieve(self, question: str, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        if self.questions is not None:
            self.questions.append(question)
        return self.results.pop(0) if self.results else []


class StubProvider(BaseLLMProvider):
    def __init__(self, answer: str = '{"query": "refined terms"}') -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, context: list[str], *, model: str | None = None) -> LLMResponse:
        _ = (context, model)
        self.prompts.append(prompt)
        return LLMResponse(self.answer, "stub", 3, 1.0, 0.0)

    def stream(
        self, prompt: str, context: list[str], *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        _ = (prompt, context, model)
        yield {"type": "final"}

    def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        return self.generate(prompt, [], model=model)

    def stream_from_prompt(
        self, prompt: str, *, model: str | None = None
    ) -> Generator[dict[str, Any]]:
        _ = (prompt, model)
        yield {"type": "final"}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def hit(
    source: str, index: int, score: float, text: str = "python retrieval terms"
) -> dict[str, Any]:
    return {"source": source, "chunk_index": index, "score": score, "text": text}


def policy(
    results: list[list[dict[str, Any]]], **kwargs: Any
) -> tuple[StubRetriever, StubProvider, Any]:
    retriever = StubRetriever(results, [])
    provider = StubProvider(kwargs.pop("provider_answer", '{"query": "refined terms"}'))
    settings = Settings(
        adaptive_enabled=True,
        adaptive_max_rounds=kwargs.pop("max_rounds", 3),
        adaptive_max_refinements=kwargs.pop("max_refinements", 1),
        adaptive_initial_top_k=2,
        adaptive_escalation_top_k=4,
        **kwargs,
    )
    return retriever, provider, AdaptiveRetrievalPolicy(settings, retriever, provider)


def test_high_evidence_answers_without_escalation_and_trace_is_serializable() -> None:
    retriever, provider, controller = policy([[hit("a.md", 0, 0.9)]])
    result = controller.run("python retrieval terms")

    assert not result.trace.abstained
    assert len(retriever.results) == 0
    assert provider.prompts == []
    assert result.trace.to_dict()["transitions"][0]["state"] == "initial_retrieve"


@pytest.mark.parametrize(
    ("results", "reason"),
    [([], StopReason.EMPTY_CORPUS), ([[]], StopReason.FILTERED_NO_RESULTS)],
)
def test_empty_and_filtered_results_abstain(
    results: list[list[dict[str, Any]]], reason: StopReason
) -> None:
    _, _, controller = policy(results)
    result = controller.run(
        "question", metadata_filter={} if reason == StopReason.EMPTY_CORPUS else {"tenant_id": "x"}
    )
    assert result.trace.stop_reason == reason


def test_escalation_then_answer_deduplicates_hits() -> None:
    first = [hit("a.md", 0, 0.1)]
    second = [hit("a.md", 0, 0.1), hit("b.md", 1, 0.8)]
    retriever, _, controller = policy([first, second])
    result = controller.run("python retrieval terms")

    assert not result.trace.abstained
    assert list(retriever.questions) == [
        "python retrieval terms",
        "python retrieval terms",
    ]
    assert [(item["source"], item["chunk_index"]) for item in result.contexts] == [
        ("a.md", 0),
        ("b.md", 1),
    ]


def test_refinement_is_retrieval_only_and_original_is_retained() -> None:
    retriever, provider, controller = policy(
        [[hit("a.md", 0, 0.1)], [hit("b.md", 1, 0.8)]], provider_answer='{"query":"refined terms"}'
    )
    result = controller.run("Original user question")

    assert retriever.questions[-1] == "refined terms"
    assert result.trace.original_query == "Original user question"
    assert provider.prompts
    assert "Original user question" in provider.prompts[0]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [("not json", StopReason.INVALID_REFINEMENT), ('{"query":""}', StopReason.INVALID_REFINEMENT)],
)
def test_invalid_refinement_abstains(answer: str, reason: StopReason) -> None:
    _, _, controller = policy(
        [[hit("a.md", 0, 0.1)], [hit("b.md", 1, 0.1)]], provider_answer=answer
    )
    assert controller.run("question").trace.stop_reason == reason


def test_provider_failure_abstains_and_repeated_evidence_is_bounded() -> None:
    _, provider, controller = policy([[hit("a.md", 0, 0.1)], [hit("b.md", 1, 0.1)]])

    def fail_generate(*_args: Any, **_kwargs: Any) -> LLMResponse:
        raise RuntimeError("down")

    provider.generate = fail_generate  # type: ignore[method-assign]
    assert controller.run("question").trace.stop_reason == StopReason.PROVIDER_FAILURE

    _, _, repeated = policy([[hit("a.md", 0, 0.1)], [hit("a.md", 0, 0.1)]], max_refinements=0)
    assert repeated.run("question").trace.stop_reason == StopReason.REPEATED_EVIDENCE


def test_round_budget_stops() -> None:
    _, _, controller = policy([[hit("a.md", 0, 0.1)], [hit("b.md", 1, 0.1)]], max_rounds=1)
    assert controller.run("question").trace.stop_reason == StopReason.INSUFFICIENT_EVIDENCE


def test_structured_critique_is_optional_observable_evidence() -> None:
    retriever, provider, controller = policy([[hit("a.md", 0, 0.9)]])
    controller.settings = controller.settings.with_overrides(adaptive_critique_enabled=True)
    provider.answer = '{"supported_claims":["claim"],"missing_evidence":[]}'

    result = controller.run("python retrieval terms")

    assert not result.trace.abstained
    evaluations = [event for event in result.trace.transitions if event.evidence]
    assert evaluations[0].evidence is not None
    assert evaluations[0].evidence.critique_supported == 1
    assert retriever.questions == ["python retrieval terms"]
