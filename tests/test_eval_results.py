from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.compare import main
from evals.results.compare import ThresholdError, compare, parse_threshold
from evals.results.schema import MetricResult, ResultError, load_result


def _result(
    tmp_path: Path, *, score: float | None = 0.8, version: int = 1, name: str = "result"
) -> Path:
    payload = {
        "schema_version": version,
        "run_id": "r",
        "timestamp": "2026-08-04T00:00:00Z",
        "dataset": {"dataset_id": "d", "dataset_version": "1", "split": "s", "checksum": "c"},
        "selected_ids": ["a"],
        "metrics": [
            {"descriptor": {"name": "quality", "direction": "higher_is_better"}, "value": score}
        ],
        "provenance": {"judge_model": "m", "embedding_model": "e", "settings_snapshot": {}},
        "status": "complete",
    }
    path = tmp_path / f"{name}-{version}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_historical_shape_migrates(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-04T00:00:00Z",
                "scores": {"quality": 0.8},
                "dataset": {
                    "dataset_id": "d",
                    "dataset_version": "1",
                    "split": "s",
                    "checksum": "c",
                    "selected_record_ids": ["a"],
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_result(path).schema_version == 1


def test_future_version_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ResultError, match="newer"):
        load_result(_result(tmp_path, version=99))


def test_compare_reports_direction_and_cases(tmp_path: Path) -> None:
    baseline = load_result(_result(tmp_path, score=0.8))
    candidate_path = _result(tmp_path, score=0.7)
    candidate = load_result(candidate_path)
    report = compare(baseline, candidate)
    assert report.deltas[0].absolute == pytest.approx(-0.1)
    assert report.regressions == ["quality"]


def test_non_finite_values_do_not_pass(tmp_path: Path) -> None:
    baseline = load_result(_result(tmp_path, score=0.8, name="baseline"))
    candidate = load_result(_result(tmp_path, score=None, name="candidate"))
    report = compare(baseline, candidate)
    assert not report.passed
    assert report.non_finite == ["quality"]


def test_metric_result_preserves_case_status_counts_and_threshold() -> None:
    result = {
        "descriptor": {
            "name": "citation_accuracy",
            "direction": "higher_is_better",
            "threshold": 0.8,
            "missing_value": "not_applicable",
        },
        "value": None,
        "cases": {"r1": None},
        "case_results": {
            "r1": {
                "value": None,
                "threshold": 0.8,
                "status": "unavailable",
                "input_ids": ["r1-c1"],
                "warning": "citation annotation is missing",
            }
        },
        "valid_count": 0,
        "missing_count": 1,
        "error_count": 0,
    }
    parsed = MetricResult.model_validate(result)
    assert parsed.case_results["r1"].status == "unavailable"
    assert parsed.descriptor.threshold == 0.8
    assert parsed.missing_count == 1


def test_threshold_grammar_and_unknown_metric() -> None:
    assert parse_threshold("quality>=0.60", {"quality"}).value == 0.6
    assert parse_threshold("quality<=2.0", {"quality"}).operator == "<="
    assert parse_threshold("quality_delta>=-0.02", {"quality"}).delta
    with pytest.raises(ThresholdError):
        parse_threshold("unknown>=1", {"quality"})


def test_cli_exit_codes(tmp_path: Path) -> None:
    baseline = _result(tmp_path, score=0.8, name="baseline")
    candidate = _result(tmp_path, score=0.7, name="candidate")
    assert main([str(candidate), "--baseline", str(baseline)]) == 1
    assert main([str(candidate), "--baseline", str(baseline), "--threshold", "bad"]) == 2
    assert main([str(candidate), "--baseline", str(tmp_path / "missing.json")]) == 2


def test_incompatible_dataset_uses_exit_code_two(tmp_path: Path) -> None:
    baseline = _result(tmp_path, name="baseline")
    candidate = _result(tmp_path, name="candidate")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["dataset"]["checksum"] = "different"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    assert main([str(candidate), "--baseline", str(baseline)]) == 2
