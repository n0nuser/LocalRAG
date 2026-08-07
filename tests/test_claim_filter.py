from __future__ import annotations

from typing import Any

from localrag.llm.types import LLMResponse
from localrag.rag.claim_filter import ClaimFilterStatus, filter_inapplicable_contexts
from localrag.settings import Settings


class _StubProvider:
    """Minimal provider double: only ``generate_from_prompt`` is exercised."""

    def __init__(self, answer: str = "", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[str] = []

    @property
    def default_model(self) -> str:
        return "stub-model"

    def generate_from_prompt(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        del model
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            answer=self.answer,
            model="stub-model",
            tokens_used=0,
            latency_ms=0.0,
            estimated_cost_usd=0.0,
        )


def _contexts() -> list[dict[str, Any]]:
    return [
        {
            "source": "book.md",
            "chunk_index": 1,
            "text": "Across years of habitual short sleep, cardiac risk rises sharply.",
            "metadata": {"heading_path": "Cancer > CARDIOVASCULAR"},
        },
        {
            "source": "book.md",
            "chunk_index": 2,
            "text": "After a single night of four hours, reaction time lengthens.",
            "metadata": {"heading_path": "Guinness > ONE NIGHT"},
        },
    ]


def _settings(**overrides: Any) -> Settings:
    base = Settings(claim_filter_enabled=True)
    return base.with_overrides(**overrides) if overrides else base


def test_disabled_by_default_returns_contexts_untouched() -> None:
    contexts = _contexts()
    provider = _StubProvider()

    result = filter_inapplicable_contexts(
        contexts, "What happens after one night?", Settings(), provider
    )

    assert result.contexts == contexts
    assert result.observation.status == ClaimFilterStatus.DISABLED
    assert provider.calls == []


def test_discards_only_the_contexts_the_model_marks_inapplicable() -> None:
    provider = _StubProvider(answer='{"inapplicable": [1]}')

    result = filter_inapplicable_contexts(
        _contexts(), "What happens after one night?", _settings(), provider
    )

    assert [context["chunk_index"] for context in result.contexts] == [2]
    assert result.observation.status == ClaimFilterStatus.FILTERED
    assert result.observation.discarded == 1
    assert len(provider.calls) == 1


def test_uses_a_single_provider_call_regardless_of_context_count() -> None:
    provider = _StubProvider(answer='{"inapplicable": []}')
    contexts = _contexts() * 6

    result = filter_inapplicable_contexts(contexts, "Q", _settings(), provider)

    assert len(provider.calls) == 1
    assert result.contexts == contexts


def test_provider_failure_degrades_to_the_unfiltered_contexts() -> None:
    contexts = _contexts()
    provider = _StubProvider(error=TimeoutError("slow"))

    result = filter_inapplicable_contexts(contexts, "Q", _settings(), provider)

    assert result.contexts == contexts
    assert result.observation.status == ClaimFilterStatus.FALLBACK
    assert result.observation.error == "TimeoutError"


def test_unparseable_output_degrades_rather_than_raising() -> None:
    provider = _StubProvider(answer="I think the second one, maybe?")
    contexts = _contexts()

    result = filter_inapplicable_contexts(contexts, "Q", _settings(), provider)

    assert result.contexts == contexts
    assert result.observation.status == ClaimFilterStatus.FALLBACK


def test_out_of_range_indices_are_ignored_rather_than_dropping_context() -> None:
    provider = _StubProvider(answer='{"inapplicable": [0, 99, -4]}')
    contexts = _contexts()

    result = filter_inapplicable_contexts(contexts, "Q", _settings(), provider)

    assert result.contexts == contexts
    assert result.observation.discarded == 0


def test_discarding_every_context_is_refused_as_an_unsafe_verdict() -> None:
    """Emptying the context set would silently turn a grounded answer into none.

    The engine already has an abstain path for genuinely insufficient evidence;
    a filter that removes everything is far more likely to be a bad verdict than
    a correct one, so it degrades instead.
    """
    provider = _StubProvider(answer='{"inapplicable": [1, 2]}')
    contexts = _contexts()

    result = filter_inapplicable_contexts(contexts, "Q", _settings(), provider)

    assert result.contexts == contexts
    assert result.observation.status == ClaimFilterStatus.FALLBACK
    assert result.observation.error == "all_contexts_discarded"


def test_empty_contexts_short_circuit_without_calling_the_provider() -> None:
    provider = _StubProvider()

    result = filter_inapplicable_contexts([], "Q", _settings(), provider)

    assert result.contexts == []
    assert provider.calls == []
    assert result.observation.status == ClaimFilterStatus.NO_CONTEXT


def test_observation_is_json_serialisable_for_the_query_trace() -> None:
    provider = _StubProvider(answer='{"inapplicable": [1]}')

    result = filter_inapplicable_contexts(_contexts(), "Q", _settings(), provider)
    payload = result.observation.to_dict()

    assert payload["status"] == "filtered"
    assert payload["discarded"] == 1
    assert payload["evaluated"] == 2


def test_prompt_includes_section_headings_so_scope_is_visible() -> None:
    """Judging applicability without the heading is judging blind (see #172)."""
    provider = _StubProvider(answer='{"inapplicable": []}')

    filter_inapplicable_contexts(_contexts(), "Q", _settings(), provider)

    assert "Guinness > ONE NIGHT" in provider.calls[0]


def test_context_text_is_bounded_before_being_sent_to_the_provider() -> None:
    provider = _StubProvider(answer='{"inapplicable": []}')
    contexts = [{"source": "a.md", "chunk_index": 1, "text": "x" * 10_000, "metadata": {}}]

    filter_inapplicable_contexts(
        contexts, "Q", _settings(claim_filter_input_max_chars=200), provider
    )

    assert len(provider.calls[0]) < 2_000
