from __future__ import annotations

import pytest

from evals.metrics import (
    MetricCase,
    aggregate_cases,
    exact_match,
    f1,
    resolve_retrieved_citations,
    score_citation_accuracy,
    score_judge_metric,
    score_retrieval_recall,
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


def test_retrieval_recall_prefers_the_id_join_when_namespaces_are_shared() -> None:
    """Offline the retrieved IDs are dataset citation IDs, so the join is exact."""
    result = score_retrieval_recall(
        ["c1", "c2"],
        ["c2", "c3"],
        {"c1": "acute passage", "c2": "chronic passage", "c3": "other passage"},
        ["chronic passage", "other passage"],
    )
    assert result.value == 0.5
    assert result.status == "complete"


def test_retrieval_recall_falls_back_to_text_when_ids_share_no_namespace() -> None:
    """A live run returns corpus chunk hashes; comparing them to citation IDs is meaningless."""
    result = score_retrieval_recall(
        ["c1", "c2"],
        ["sha256:aaaa", "sha256:bbbb"],
        {"c1": "the clamp cools for ninety seconds", "c2": "calibration drift accumulates"},
        ["Section 4. The clamp cools for ninety seconds, then readings resume."],
    )
    assert result.value == 0.5


def test_retrieval_recall_matches_a_citation_inside_a_larger_chunk() -> None:
    """Chunk boundaries cut passages, so exact containment alone would under-count."""
    hits = resolve_retrieved_citations(
        ["c1"],
        None,
        {"c1": "readings are unavailable for approximately ninety seconds"},
        ["Readings are unavailable, in practice, for ninety seconds or so while it cools."],
    )
    assert hits == {"c1"}


def test_retrieval_recall_is_unavailable_rather_than_zero_without_annotations() -> None:
    result = score_retrieval_recall(None, ["c1"], {"c1": "text"}, ["text"])
    assert result.status == "unavailable"
    assert result.value is None


def test_retrieval_recall_is_unavailable_when_neither_join_can_be_made() -> None:
    """No shared IDs and no retrieved text is undecidable, not a score of zero."""
    result = score_retrieval_recall(["c1"], ["sha256:aaaa"], {"c1": "text"}, [])
    assert result.status == "unavailable"
    assert result.value is None
