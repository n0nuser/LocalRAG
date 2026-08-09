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


#: Fraction of a citation's tokens that must appear in a retrieved context for the
#: text join to count it as retrieved. Deliberately below 1.0: chunk boundaries cut
#: passages, so an annotated citation is often a subset of a larger retrieved chunk.
RETRIEVAL_RECALL_TOKEN_COVERAGE = 0.6


def _context_contains_citation(citation_text: str, contexts: list[str]) -> bool:
    """Whether any retrieved context carries this citation's passage."""
    needle = normalize_answer(citation_text)
    if not needle:
        return False
    needle_tokens = set(needle.split())
    for context in contexts:
        haystack = normalize_answer(context)
        if needle in haystack:
            return True
        if needle_tokens:
            covered = len(needle_tokens & set(haystack.split())) / len(needle_tokens)
            if covered >= RETRIEVAL_RECALL_TOKEN_COVERAGE:
                return True
    return False


def resolve_retrieved_citations(
    relevant_ids: list[str] | None,
    retrieved_ids: list[str] | None,
    citation_texts: dict[str, str] | None,
    retrieved_texts: list[str] | None,
) -> set[str] | None:
    """Return which relevant citations retrieval actually surfaced.

    ``None`` means undecidable — the caller must not read that as "none were
    retrieved". This is the single join both ``retrieval_recall`` and the
    ``context_omission`` failure label use, so they can never disagree.

    Two joins, because the two run modes name chunks differently. When the
    retrieved IDs live in the dataset's own citation namespace the ID join is
    exact and is preferred. A live run returns corpus chunk hashes instead,
    which share no namespace with dataset citation IDs, so comparing them is
    meaningless; there the citation's *text* is matched against the retrieved
    context text.
    """
    if not relevant_ids:
        return None
    known = citation_texts or {}
    relevant = set(relevant_ids)
    # Overlap with the declared citation IDs is what proves a shared namespace.
    # With no citation text supplied there is nothing to prove it against and
    # nothing to fall back to, so the ID join is taken at face value.
    if retrieved_ids and (not known or set(retrieved_ids) & set(known)):
        return relevant & set(retrieved_ids)
    if not retrieved_texts:
        return None
    if any(not known.get(citation_id) for citation_id in relevant):
        return None
    return {
        citation_id
        for citation_id in relevant
        if _context_contains_citation(known[citation_id], retrieved_texts)
    }


def score_retrieval_recall(
    relevant_ids: list[str] | None,
    retrieved_ids: list[str] | None,
    citation_texts: dict[str, str] | None,
    retrieved_texts: list[str] | None,
) -> MetricCase:
    """Score the fraction of annotated-relevant citations that were retrieved.

    Complements ``citation_accuracy``, which is set *precision* over what the
    answer cited. Neither could previously catch retrieval silently returning
    topically-similar passages instead of the ones that answer the question.
    """
    hits = resolve_retrieved_citations(relevant_ids, retrieved_ids, citation_texts, retrieved_texts)
    if hits is None:
        return MetricCase(status="unavailable", warning="retrieval recall join is unavailable")
    return MetricCase(value=len(hits) / len(set(relevant_ids or [])))
