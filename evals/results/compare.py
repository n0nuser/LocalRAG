"""Direction-aware comparison and threshold parsing for benchmark results."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from evals.results.schema import ResultFile

THRESHOLD_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_.-]*)(?P<op>>=|<=)(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+))$"
)


class ThresholdError(ValueError):
    """A threshold expression is malformed or refers to an unknown metric."""


@dataclass(frozen=True)
class Threshold:
    name: str
    operator: str
    value: float
    delta: bool = False


@dataclass
class MetricDelta:
    name: str
    absolute: float | None
    relative: float | None
    regression: bool = False
    non_finite: bool = False


@dataclass
class ComparisonReport:
    comparable: bool
    regressions: list[str] = field(default_factory=list)
    added_metrics: list[str] = field(default_factory=list)
    removed_metrics: list[str] = field(default_factory=list)
    missing_cases: list[str] = field(default_factory=list)
    extra_cases: list[str] = field(default_factory=list)
    non_finite: list[str] = field(default_factory=list)
    deltas: list[MetricDelta] = field(default_factory=list)
    incompatibilities: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.comparable
            and not self.regressions
            and not self.added_metrics
            and not self.removed_metrics
            and not self.non_finite
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "comparable": self.comparable,
            "passed": self.passed,
            "regressions": self.regressions,
            "added_metrics": self.added_metrics,
            "removed_metrics": self.removed_metrics,
            "missing_cases": self.missing_cases,
            "extra_cases": self.extra_cases,
            "non_finite": self.non_finite,
            "incompatibilities": self.incompatibilities,
            "deltas": [delta.__dict__ for delta in self.deltas],
        }


def parse_threshold(expression: str, metric_names: set[str]) -> Threshold:
    match = THRESHOLD_RE.fullmatch(expression.strip())
    if not match:
        message = f"malformed threshold {expression!r}; use metric>=0.60 or metric<=2.0"
        raise ThresholdError(message)
    expression_name = match.group("name")
    delta = expression_name.endswith("_delta")
    name = expression_name.removesuffix("_delta") if delta else expression_name
    if name not in metric_names:
        message = f"unknown metric in threshold: {name!r}"
        raise ThresholdError(message)
    return Threshold(name, match.group("op"), float(match.group("value")), delta)


def compare(  # noqa: C901, PLR0912
    baseline: ResultFile, candidate: ResultFile, thresholds: list[Threshold] | None = None
) -> ComparisonReport:
    report = ComparisonReport(comparable=True)
    if baseline.dataset != candidate.dataset:
        report.comparable = False
        report.incompatibilities.append("dataset identity/checksum differs")
    for key in ("judge_model", "embedding_model", "settings_snapshot"):
        if (
            key in baseline.provenance
            and key in candidate.provenance
            and baseline.provenance[key] != candidate.provenance[key]
        ):
            report.comparable = False
            report.incompatibilities.append(f"provenance {key} differs")
    old, new = baseline.metric_map(), candidate.metric_map()
    report.added_metrics = sorted(set(new) - set(old))
    report.removed_metrics = sorted(set(old) - set(new))
    for name in sorted(set(old) & set(new)):
        if old[name].descriptor != new[name].descriptor:
            report.comparable = False
            report.incompatibilities.append(f"metric descriptor {name} differs")
        before, after = old[name].value, new[name].value
        if before is None or after is None:
            report.non_finite.append(name)
            continue
        absolute = after - before
        relative = absolute / abs(before) if before else None
        direction = new[name].descriptor.direction
        regression = absolute < 0 if direction == "higher_is_better" else absolute > 0
        report.deltas.append(MetricDelta(name, absolute, relative, regression))
        if regression:
            report.regressions.append(name)
    report.missing_cases = sorted(set(baseline.selected_ids) - set(candidate.selected_ids))
    report.extra_cases = sorted(set(candidate.selected_ids) - set(baseline.selected_ids))
    if report.missing_cases or report.extra_cases:
        report.comparable = False
        report.incompatibilities.append("selected case IDs differ")
    for threshold in thresholds or []:
        metric = new[threshold.name]
        value = None
        if threshold.delta:
            delta = next((item for item in report.deltas if item.name == threshold.name), None)
            value = delta.absolute if delta else None
        else:
            value = metric.value
        if value is None or not math.isfinite(value):
            report.regressions.append(f"{threshold.name}: missing/non-finite threshold value")
        elif (threshold.operator == ">=" and value < threshold.value) or (
            threshold.operator == "<=" and value > threshold.value
        ):
            report.regressions.append(
                f"{threshold.name}: threshold {threshold.operator}{threshold.value} failed"
            )
    return report
