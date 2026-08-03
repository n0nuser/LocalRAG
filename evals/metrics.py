"""Pinned local metric contracts used by the evaluation runner.

The deterministic metrics intentionally do not depend on RAGAS or an LLM.
Normalization is Unicode lower-casing, punctuation removal, and whitespace
collapse; tokenization is the resulting whitespace-separated words.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class MetricCase:
    """A metric result for one record."""

    value: float | None = None
    status: str = "complete"
    error: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class MetricAggregate:
    """Aggregate metric result with explicit missing/error accounting."""

    value: float | None
    valid_count: int
    missing_count: int
    error_count: int


def normalize_answer(value: str) -> str:
    """Normalize answer text for EM/F1; this implementation is version-pinned."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def _tokens(value: str) -> list[str]:
    return normalize_answer(value).split()


def exact_match(prediction: str, references: list[str]) -> float:
    """Return one if normalized prediction exactly matches any reference."""
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(reference) for reference in references)
    )


def _f1_one(prediction: str, reference: str) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def f1(prediction: str, references: list[str]) -> float:
    """Return maximum token-multiset F1 over the provided references."""
    return max((_f1_one(prediction, reference) for reference in references), default=0.0)


def aggregate_cases(cases: list[MetricCase]) -> MetricAggregate:
    valid = [case.value for case in cases if case.status == "complete" and case.value is not None]
    errors = sum(case.status == "error" for case in cases)
    missing = len(cases) - len(valid) - errors
    return MetricAggregate(
        value=sum(valid) / len(valid) if valid else None,
        valid_count=len(valid),
        missing_count=missing,
        error_count=errors,
    )


def score_judge_metric(judge: Callable[[], float]) -> MetricCase:
    """Adapt a judge call, treating exceptions and NaN as explicit errors."""
    try:
        value = float(judge())
    except Exception as exc:
        return MetricCase(status="error", error=str(exc))
    if not math.isfinite(value):
        return MetricCase(status="error", error="judge returned a non-finite value")
    return MetricCase(value=value)


def score_citation_accuracy(
    answer: str, cited_ids: list[str] | None, relevant_ids: list[str] | None
) -> MetricCase:
    """Score cited IDs against annotation-backed relevant IDs.

    ``answer`` is accepted as part of the input contract for future span-aware
    implementations. This version scores set precision over annotated IDs.
    Missing annotations are not a zero or perfect score.
    """
    del answer
    if not cited_ids or not relevant_ids:
        return MetricCase(status="unavailable", warning="citation annotation is missing")
    relevant = set(relevant_ids)
    return MetricCase(value=len(set(cited_ids) & relevant) / len(set(cited_ids)))
