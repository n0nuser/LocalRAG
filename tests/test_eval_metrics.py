from __future__ import annotations

import pytest

from evals.metrics import (
    MetricCase,
    aggregate_cases,
    exact_match,
    f1,
    score_citation_accuracy,
    score_judge_metric,
)


@pytest.mark.parametrize(
    ("prediction", "references", "expected"),
    [
        ("The Answer!", ["the answer"], 1.0),
        ("", [""], 1.0),
        ("", ["something"], 0.0),
        ("wrong", ["the answer", "wrong"], 1.0),
    ],
)
def test_exact_match_uses_pinned_normalization_and_multiple_references(
    prediction: str, references: list[str], expected: float
) -> None:
    assert exact_match(prediction, references) == expected


def test_f1_is_token_multiset_f1() -> None:
    assert f1("a a b", ["a b c"]) == pytest.approx(2 / 3)


def test_empty_f1_and_abstention_are_explicit() -> None:
    assert f1("", [""]) == 1.0
    assert f1("", ["answer"]) == 0.0


def test_aggregate_ignores_unavailable_and_records_counts() -> None:
    aggregate = aggregate_cases(
        [
            MetricCase(value=1.0),
            MetricCase(status="unavailable"),
            MetricCase(status="error", error="boom"),
        ]
    )
    assert aggregate.value == 1.0
    assert aggregate.valid_count == 1
    assert aggregate.missing_count == 1
    assert aggregate.error_count == 1


def test_judge_metric_records_errors_and_non_finite_values() -> None:
    result = score_judge_metric(lambda: float("nan"))
    assert result.status == "error"
    assert result.value is None
    assert result.error == "judge returned a non-finite value"


def test_citation_accuracy_is_unavailable_without_annotations() -> None:
    result = score_citation_accuracy("answer", [], ["c1"])
    assert result.status == "unavailable"
    assert result.value is None
    assert "annotation" in (result.warning or "")


def test_citation_accuracy_scores_annotation_ids() -> None:
    result = score_citation_accuracy("answer", ["c1", "missing"], ["c1", "c2"])
    assert result.value == 0.5
    assert result.status == "complete"
