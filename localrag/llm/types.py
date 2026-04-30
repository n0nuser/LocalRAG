"""Shared data types for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    answer: str
    model: str
    tokens_used: int
    latency_ms: float
    estimated_cost_usd: float
    sources: list[dict[str, object]] = field(default_factory=list)
