from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.matrix import (
    MatrixConfig,
    MatrixValidationError,
    expand_matrix,
    run_matrix,
)
from evals.tracking import TrackingConfig, TrackingSession


def _config(**dimensions: object) -> MatrixConfig:
    return MatrixConfig(
        matrix_id="fixture",
        profile="fixture",
        dataset={
            "dataset_id": "localrag-core",
            "dataset_version": "1.0.0",
            "split": "smoke",
            "checksum": "abc",
        },
        dimensions=dimensions or {"retrieval_mode": ["vector", "hybrid"]},
        seed=7,
    )


def test_matrix_schema_rejects_unknown_fields_and_bad_version() -> None:
    with pytest.raises(ValidationError):
        MatrixConfig.model_validate(
            {"matrix_id": "x", "profile": "fixture", "dimensions": {}, "extra": 1}
        )
    with pytest.raises(ValidationError, match="schema_version"):
        MatrixConfig.model_validate(
            {"schema_version": 99, "matrix_id": "x", "profile": "fixture", "dimensions": {}}
        )


def test_expansion_is_cartesian_sorted_and_ids_ignore_input_order() -> None:
    first = _config(retrieval_mode=["hybrid", "vector"], chunking_mode=["fixed", "structural"])
    second = _config(chunking_mode=["structural", "fixed"], retrieval_mode=["vector", "hybrid"])
    a = expand_matrix(first)
    b = expand_matrix(second)
    assert [(case.case_id, case.effective_config) for case in a] == [
        (case.case_id, case.effective_config) for case in b
    ]
    assert len(a) == 4
    assert [case.effective_config for case in a] == [
        {"chunking_mode": "fixed", "retrieval_mode": "hybrid"},
        {"chunking_mode": "fixed", "retrieval_mode": "vector"},
        {"chunking_mode": "structural", "retrieval_mode": "hybrid"},
        {"chunking_mode": "structural", "retrieval_mode": "vector"},
    ]


def test_unsupported_and_incompatible_combinations_fail_before_execution() -> None:
    with pytest.raises(MatrixValidationError, match="unsupported dimension"):
        expand_matrix(_config(not_a_dimension=["x"]))
    with pytest.raises(MatrixValidationError, match="not supported"):
        expand_matrix(_config(embedding_model=["e5-large-v2"]))
    with pytest.raises(MatrixValidationError, match="reranking"):
        expand_matrix(_config(rerank_enabled=[True]))


def test_dry_run_writes_ordered_cases_without_executing(tmp_path: Path) -> None:
    calls: list[str] = []
    manifest = run_matrix(
        _config(), tmp_path, dry_run=True, executor=lambda case, _work: calls.append(case.case_id)
    )
    assert calls == []
    assert [case["case_id"] for case in manifest["cases"]] == [
        manifest["cases"][0]["case_id"],
        manifest["cases"][1]["case_id"],
    ]
    assert (tmp_path / "fixture" / "manifest.json").exists()
    assert manifest["status"] == "dry_run"


def test_runner_isolates_cases_and_continues_after_failure(tmp_path: Path) -> None:
    def execute(case: object, work: Path) -> dict[str, object]:
        work.joinpath("marker").write_text("only this case", encoding="utf-8")
        if case.effective_config["retrieval_mode"] == "vector":
            raise RuntimeError("fixture exploded")
        return {"metrics": {"quality": 1.0}, "artifacts": ["marker"]}

    manifest = run_matrix(_config(), tmp_path, executor=execute)
    assert manifest["status"] == "failed"
    assert [case["status"] for case in manifest["cases"]] == ["complete", "failed"]
    assert manifest["exit_code"] == 1
    for case in manifest["cases"]:
        assert (tmp_path / "fixture" / "cases" / case["case_id"] / "marker").exists() or case[
            "status"
        ] == "failed"
        assert case["run_id"]
    loaded = json.loads((tmp_path / "fixture" / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["matrix_id"] == "fixture"


def test_tracking_receives_stable_parent_and_nested_case_ids(tmp_path: Path) -> None:
    class Backend:
        def __init__(self) -> None:
            self.started: list[tuple[str, str]] = []

        def start_parent(self, run_id: str, params: dict[str, object]) -> None:
            self.started.append(("parent", run_id))

        def start_case(self, case_id: str, params: dict[str, object]) -> None:
            self.started.append(("case", case_id))

        def log_metrics(self, metrics: dict[str, float]) -> None: ...

        def log_artifact(self, path: Path) -> None: ...

        def finish_case(self, status: str, error: str | None) -> None: ...

        def finish_parent(self, status: str, error: str | None) -> None: ...

        def close(self) -> None: ...

    backend = Backend()
    tracker = TrackingSession(TrackingConfig(enabled=True), backend=backend)
    first = run_matrix(_config(), tmp_path / "first", tracker=tracker)
    second = run_matrix(_config(), tmp_path / "second", tracker=tracker)

    assert backend.started == [
        ("parent", first["run_id"]),
        *[("case", case["case_id"]) for case in first["cases"]],
        ("parent", second["run_id"]),
        *[("case", case["case_id"]) for case in second["cases"]],
    ]
