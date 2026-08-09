"""Deterministic, privacy-preserving analysis of canonical evaluation cases.

This module consumes the per-case inputs already assembled by the evaluation
runner. It does not score metrics or replace RAGAS; it explains failed cases.
"""

from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals.metrics import resolve_retrieved_citations

FAILURE_LABELS = (
    "retrieval_miss",
    "context_omission",
    "unsupported_claim",
    "wrong_citation",
    "out_of_scope_refusal",
    "evaluator_failure",
    "unclassified",
)
FailureLabel = Literal[
    "retrieval_miss",
    "context_omission",
    "unsupported_claim",
    "wrong_citation",
    "out_of_scope_refusal",
    "evaluator_failure",
    "unclassified",
]
_ORDER = {label: index for index, label in enumerate(FAILURE_LABELS)}
_REFUSALS = (
    "i don't know",
    "cannot answer",
    "can't answer",
    "outside my scope",
    "not able to answer",
)


class FailureCaseArtifact(BaseModel):
    """Inputs available before classification; text is never emitted in output."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: str = "completed"
    answer: str | None = None
    ground_truth: str | None = None
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieved_text: list[str] = Field(default_factory=list)
    citation_ids: list[str] | None = None
    relevant_citation_ids: list[str] | None = None
    citation_texts: dict[str, str] | None = Field(
        default=None,
        description="Declared citation ID to passage; lets context_omission join on text.",
    )
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("metrics")
    @classmethod
    def _metric_objects(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return value


class FailureAnalysisConfig(BaseModel):
    """Explicit opt-in configuration for the optional local judge."""

    judge_enabled: bool = False
    judge: Callable[[FailureCaseArtifact], Any] | None = Field(default=None, exclude=True)
    judge_model: str | None = None
    judge_timeout_seconds: float = 5.0
    judge_retries: int = 1

    model_config = ConfigDict(arbitrary_types_allowed=True)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[FailureLabel]
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    rationale_codes: list[str] = Field(default_factory=list)


class FailureClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    failed: bool
    labels: list[FailureLabel]
    primary_label: FailureLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    classifier_version: str = "failure-taxonomy-v1"
    judge_model: str | None = None
    redacted: bool = True


class FailureAnalysisSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[FailureClassification]
    counts: dict[str, int]
    failed_count: int
    classifier_version: str = "failure-taxonomy-v1"


def _metric_failed(metric: dict[str, Any]) -> bool:
    if metric.get("status") in {"error", "unavailable"}:
        return metric.get("status") == "error"
    value = metric.get("value")
    threshold = metric.get("threshold")
    if value is None or threshold is None:
        return False
    try:
        value_float = float(value)
        threshold_float = float(threshold)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value_float):
        return metric.get("status") == "error"
    direction = metric.get("direction", "higher_is_better")
    return (
        value_float < threshold_float
        if direction == "higher_is_better"
        else value_float > threshold_float
    )


def _judge(artifact: FailureCaseArtifact, config: FailureAnalysisConfig) -> JudgeVerdict | None:
    if not config.judge_enabled or config.judge is None:
        return None
    for _ in range(config.judge_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw = pool.submit(config.judge, artifact).result(
                    timeout=config.judge_timeout_seconds
                )
            verdict = JudgeVerdict.model_validate(
                {**raw, "model": raw.get("model", config.judge_model or "unknown")}
            )
            if not verdict.labels:
                raise ValueError("judge returned no labels")  # noqa: TRY301
            return verdict  # noqa: TRY300
        except Exception:  # noqa: S112
            continue
    return None


def _omits_relevant_context(artifact: FailureCaseArtifact) -> bool:
    """Whether annotated-relevant context was left out of what retrieval returned.

    Joining relevant IDs against retrieved IDs directly cannot work in either
    mode: offline the two sets were the same list by construction, so the
    difference was always empty and this label could never fire; live they are
    dataset citation IDs versus corpus chunk hashes, sharing no namespace, so it
    fired for every record. ``resolve_retrieved_citations`` picks the join that
    means something in each mode and reports ``None`` when neither does — which
    is not evidence of omission, so it must not be labelled as one.
    """
    retrieved_citations = resolve_retrieved_citations(
        artifact.relevant_citation_ids,
        artifact.retrieved_ids,
        artifact.citation_texts,
        artifact.retrieved_text,
    )
    if retrieved_citations is None:
        return False
    return bool(set(artifact.relevant_citation_ids or []) - retrieved_citations)


def classify_case(  # noqa: C901
    artifact: FailureCaseArtifact, *, config: FailureAnalysisConfig | None = None
) -> FailureClassification:
    """Classify one artifact using ordered deterministic evidence, then an opt-in judge."""
    config = config or FailureAnalysisConfig()
    reasons: list[str] = []
    labels: set[str] = set()
    metric_failures = [name for name, metric in artifact.metrics.items() if _metric_failed(metric)]
    failed = artifact.status != "completed" or bool(artifact.error) or bool(metric_failures)

    if (
        artifact.status != "completed"
        or artifact.error
        or any(metric.get("status") == "error" for metric in artifact.metrics.values())
    ):
        labels.add("evaluator_failure")
        reasons.append("evaluator_failure")
    if not artifact.retrieved_ids and not artifact.retrieved_text:
        labels.add("retrieval_miss")
        reasons.append("no_retrieved_context")
        failed = True
    if _omits_relevant_context(artifact):
        labels.add("context_omission")
        reasons.append("relevant_context_not_retrieved")
        failed = True
    if (
        artifact.citation_ids
        and artifact.retrieved_ids
        and not set(artifact.citation_ids).issubset(set(artifact.retrieved_ids))
    ):
        labels.add("wrong_citation")
        reasons.append("citation_not_in_retrieved_context")
        failed = True
    faithfulness = artifact.metrics.get("faithfulness", {})
    hallucination = artifact.metrics.get("hallucination_rate", {})
    if not any(metric.get("status") == "error" for metric in (faithfulness, hallucination)) and (
        _metric_failed(faithfulness) or _metric_failed(hallucination)
    ):
        labels.add("unsupported_claim")
        reasons.append("unsupported_claim_metric")
        failed = True
    if artifact.answer and any(phrase in artifact.answer.casefold() for phrase in _REFUSALS):
        labels.add("out_of_scope_refusal")
        reasons.append("refusal_language")
        failed = True
    elif artifact.answer == "":
        reasons.append("empty_answer")
        failed = True

    judge = _judge(artifact, config) if config.judge_enabled and not labels else None
    if config.judge_enabled and judge is None and config.judge is not None and not labels:
        reasons.append("invalid_judge_output")
    if judge is not None:
        labels.update(judge.labels)
        reasons.extend(judge.rationale_codes)
    if not labels:
        labels.add("unclassified")
        reasons.append("insufficient_evidence")
    if "unclassified" in labels and len(labels) > 1:
        labels.remove("unclassified")
    ordered = sorted(labels, key=lambda label: _ORDER[label])
    typed_labels = [cast("FailureLabel", label) for label in ordered]
    confidence = 1.0 if len(ordered) == 1 and ordered[0] != "unclassified" else 0.5
    if judge is not None:
        confidence = judge.confidence
    return FailureClassification(
        case_id=artifact.case_id,
        failed=failed,
        labels=typed_labels,
        primary_label=typed_labels[0],
        confidence=confidence,
        reasons=sorted(set(reasons)),
        judge_model=judge.model if judge else None,
    )


def classify_cases(
    artifacts: Iterable[FailureCaseArtifact], *, config: FailureAnalysisConfig | None = None
) -> FailureAnalysisSummary:
    cases = sorted(
        (classify_case(item, config=config) for item in artifacts), key=lambda item: item.case_id
    )
    counts = {label: sum(label in case.labels for case in cases) for label in FAILURE_LABELS}
    return FailureAnalysisSummary(
        cases=cases, counts=counts, failed_count=sum(case.failed for case in cases)
    )
