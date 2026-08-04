from __future__ import annotations

import math

from evals.failure_analysis import (
    FailureAnalysisConfig,
    FailureCaseArtifact,
    classify_case,
    classify_cases,
)


def artifact(**overrides: object) -> FailureCaseArtifact:
    values: dict[str, object] = {
        "case_id": "case-1",
        "status": "completed",
        "answer": "The answer",
        "ground_truth": "The answer",
        "retrieved_ids": ["c1"],
        "retrieved_text": ["evidence"],
        "citation_ids": ["c1"],
        "relevant_citation_ids": ["c1"],
        "metrics": {"exact_match": {"value": 1.0, "status": "complete"}},
    }
    values.update(overrides)
    return FailureCaseArtifact.model_validate(values)


def test_retrieval_miss_has_deterministic_primary_label() -> None:
    result = classify_case(artifact(retrieved_ids=[], retrieved_text=[]))
    assert result.failed is True
    assert result.primary_label == "retrieval_miss"
    assert result.labels == ["retrieval_miss"]


def test_omission_and_wrong_citation_are_multilabel_failures() -> None:
    result = classify_case(
        artifact(
            retrieved_ids=["c1"],
            relevant_citation_ids=["c1", "c2"],
            citation_ids=["c2"],
            metrics={
                "exact_match": {"value": 0.0, "status": "complete"},
                "citation_accuracy": {"value": 0.0, "status": "complete", "threshold": 0.8},
            },
        )
    )
    assert result.labels == ["context_omission", "wrong_citation"]
    assert result.primary_label == "context_omission"


def test_missing_citation_annotations_are_unclassified_not_zero() -> None:
    result = classify_case(
        artifact(
            citation_ids=None,
            relevant_citation_ids=None,
            metrics={"citation_accuracy": {"value": None, "status": "unavailable"}},
        )
    )
    assert result.labels == ["unclassified"]
    assert result.confidence < 1


def test_non_finite_metric_stays_unclassified() -> None:
    result = classify_case(
        artifact(metrics={"faithfulness": {"value": None, "status": "unavailable"}})
    )
    assert result.labels == ["unclassified"]


def test_empty_answer_is_not_invented_as_hallucination() -> None:
    result = classify_case(artifact(answer="", ground_truth="answer"))
    assert result.labels == ["unclassified"]
    assert "empty_answer" in result.reasons


def test_evaluator_failure_precedes_other_labels() -> None:
    result = classify_case(
        artifact(
            status="failed",
            error="judge timeout",
            metrics={"faithfulness": {"value": math.nan, "status": "error"}},
        )
    )
    assert result.primary_label == "evaluator_failure"
    assert "evaluator_failure" in result.labels


def test_refusal_and_unsupported_claim_can_coexist() -> None:
    result = classify_case(
        artifact(
            answer="I don't know; the claim is unsupported.",
            metrics={"faithfulness": {"value": 0.1, "status": "complete", "threshold": 0.6}},
        )
    )
    assert result.labels == ["unsupported_claim", "out_of_scope_refusal"]


def test_optional_judge_is_bounded_and_invalid_output_is_unclassified() -> None:
    result = classify_case(
        artifact(answer="ambiguous"),
        config=FailureAnalysisConfig(judge_enabled=True, judge=lambda _: {"labels": ["nope"]}),
    )
    assert result.labels == ["unclassified"]
    assert "invalid_judge_output" in result.reasons
    assert result.judge_model is None


def test_repeated_classification_and_summary_order_are_stable() -> None:
    cases = [
        artifact(case_id="b", retrieved_ids=[], retrieved_text=[]),
        artifact(case_id="a", retrieved_ids=[], retrieved_text=[]),
    ]
    first = classify_cases(cases)
    second = classify_cases(cases)
    assert first == second
    assert [item.case_id for item in first.cases] == ["a", "b"]
    assert first.failed_count == 2


def test_redaction_does_not_export_answer_or_context() -> None:
    result = classify_case(artifact(answer="SECRET DOCUMENT", retrieved_text=["PRIVATE CONTENT"]))
    payload = result.model_dump_json()
    assert "SECRET DOCUMENT" not in payload
    assert "PRIVATE CONTENT" not in payload
