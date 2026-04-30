from __future__ import annotations

import pytest

from localrag.llm.costs import estimate_cost_usd


@pytest.mark.parametrize(
    ("model", "tokens", "expected_min", "expected_max"),
    [
        ("gpt-4o-mini", 1000, 0.0001, 0.001),
        ("gpt-4o", 1000, 0.001, 0.01),
        ("claude-haiku-4-5", 1000, 0.0001, 0.001),
        ("_default_ollama", 1000, 0.0, 0.0),
    ],
)
def test_estimate_cost_known_models(
    model: str, tokens: int, expected_min: float, expected_max: float
) -> None:
    cost = estimate_cost_usd(model, tokens)
    assert expected_min <= cost <= expected_max


def test_estimate_cost_zero_tokens() -> None:
    assert estimate_cost_usd("gpt-4o", 0) == 0.0


def test_estimate_cost_unknown_model_uses_fallback() -> None:
    cost = estimate_cost_usd("some-new-mystery-model", 1000)
    assert cost > 0.0


def test_estimate_cost_prefix_match() -> None:
    cost_exact = estimate_cost_usd("gpt-4o-mini", 500)
    cost_prefix = estimate_cost_usd("gpt-4o-mini-2025-01", 500)
    assert cost_exact == cost_prefix
