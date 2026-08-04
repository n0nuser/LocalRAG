from __future__ import annotations

from typing import Any

import pytest

from evals.long_context import (
    CapabilityStatus,
    ContextBudgetError,
    build_context,
    generate_live_answer,
    probe_model_capability,
    validate_context_window,
)


def test_context_budget_orders_chunks_and_reserves_prompt_and_output() -> None:
    plan = build_context(
        "alpha",
        [
            {"id": "b", "text": "alpha beta", "score": 0.9},
            {"id": "a", "text": "alpha", "score": 0.9},
            {"id": "c", "text": "distractor words fill the remaining budget", "score": 0.1},
        ],
        window=10,
        prompt_overhead_tokens=2,
        output_tokens=3,
        strategy="stuff",
    )

    assert [chunk["id"] for chunk in plan.chunks] == ["a", "b"]
    assert plan.context_tokens == 3
    assert plan.input_tokens == 5
    assert plan.truncated is True
    assert plan.budget_tokens == 5


def test_fixed_top_k_deduplicates_and_does_not_silently_truncate() -> None:
    plan = build_context(
        "question",
        [{"id": "a", "text": "one two three", "score": 1.0}] * 2,
        window=20,
        prompt_overhead_tokens=1,
        output_tokens=2,
        strategy="fixed_top_k",
        top_k=1,
    )
    assert [chunk["id"] for chunk in plan.chunks] == ["a"]
    assert plan.truncated is False


def test_token_budget_can_fail_explicitly() -> None:
    with pytest.raises(ContextBudgetError):
        build_context(
            "a", [{"id": "a", "text": "one"}], window=2, prompt_overhead_tokens=1, output_tokens=1
        )


def test_capability_probe_reads_digest_and_native_window() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "details": {"digest": "sha256:abc"},
                "model_info": {"llama.context_length": 8192},
            }

    class Client:
        def post(self, *_: Any, **__: Any) -> Response:
            return Response()

    capability = probe_model_capability(Client(), "gemma3:4b")
    assert capability.status is CapabilityStatus.AVAILABLE
    assert capability.native_context_window == 8192
    assert capability.digest == "sha256:abc"


def test_unsupported_window_has_reason() -> None:
    result = validate_context_window(32768, native_context_window=8192)
    assert result.status is CapabilityStatus.UNSUPPORTED
    assert "8192" in result.reason


def test_mocked_live_generation_records_answer_tokens() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"message": {"content": "alpha answer"}}

    class Client:
        def post(self, *_: Any, **__: Any) -> Response:
            return Response()

    answer, timing = generate_live_answer(
        Client(),
        model="gemma3:4b",
        question="alpha?",
        context=build_context(
            "alpha",
            [{"id": "a", "text": "alpha"}],
            window=16,
            prompt_overhead_tokens=1,
            output_tokens=4,
        ),
        timeout_seconds=1,
        seed=42,
    )
    assert answer == "alpha answer"
    assert timing["output_tokens"] == 2


def test_live_generation_failure_is_not_converted_to_quality() -> None:
    class Client:
        def post(self, *_: Any, **__: Any) -> Any:
            raise TimeoutError("model timed out")

    with pytest.raises(TimeoutError, match="timed out"):
        generate_live_answer(
            Client(),
            model="gemma3:4b",
            question="alpha?",
            context=build_context(
                "alpha",
                [{"id": "a", "text": "alpha"}],
                window=16,
                prompt_overhead_tokens=1,
                output_tokens=4,
            ),
            timeout_seconds=1,
            seed=42,
        )
