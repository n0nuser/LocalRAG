"""Canonical benchmark matrix contract and runner.

The matrix runner deliberately owns orchestration only. A case executor is an
adapter for the existing evaluator (or a future profile), which keeps the
contract testable and prevents benchmark cases from sharing mutable state.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals.tracking import TrackingSession
from localrag.observability.tracing import SpanName, span

MATRIX_SCHEMA_VERSION = 1
SUPPORTED_DIMENSIONS: dict[str, tuple[Any, ...]] = {
    "provider": ("ollama",),
    "embedding_model": ("nomic-embed-text",),
    "generation_model": ("gemma3:4b",),
    "retrieval_mode": ("hybrid", "vector"),
    "retrieval_experiment_mode": ("baseline", "rewrite", "hyde", "rewrite+hyde"),
    "chunking_mode": ("fixed", "structural"),
    "rerank_enabled": (False,),
    "context_window": ("default", 4096, 8192, 32768),
    "context_strategy": ("fixed_top_k", "stuff"),
    "top_k": (5,),
    "metric_profile": ("default",),
}


class MatrixValidationError(ValueError):
    """The matrix cannot be expanded into executable cases."""


class DatasetReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_version: str
    split: str
    checksum: str


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = MATRIX_SCHEMA_VERSION
    matrix_id: str
    profile: str
    dataset: DatasetReference
    dimensions: dict[str, list[Any]] = Field(default_factory=dict)
    seed: int = 42
    mode: str = "offline"

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        if value not in {"offline", "fixture-offline", "live-local"}:
            raise ValueError("mode must be offline, fixture-offline, or live-local")
        return value

    @field_validator("schema_version")
    @classmethod
    def _version(cls, value: int) -> int:
        if value != MATRIX_SCHEMA_VERSION:
            message = f"schema_version {value} is not supported"
            raise ValueError(message)
        return value

    @field_validator("seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value


class ExpandedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    effective_config: dict[str, Any]


class MatrixCaseResult(BaseModel):
    """Versioned per-case outcome written to a matrix manifest."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    run_id: str
    matrix_id: str
    dataset: DatasetReference
    corpus: dict[str, str]
    code_revision: str
    code_dirty: bool
    model: dict[str, str]
    supported_dimensions: dict[str, tuple[Any, ...]]
    seed: int
    effective_config: dict[str, Any]
    mode: str = "offline"
    status: str
    started_at: str
    finished_at: str
    metrics: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, Any]
    resources: dict[str, Any]
    error: dict[str, str] | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class MatrixManifest(BaseModel):
    """Versioned cross-case manifest and provenance contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    matrix_id: str
    run_id: str
    profile: str
    dataset: DatasetReference
    corpus: dict[str, str]
    code_revision: str
    code_dirty: bool
    model: dict[str, str]
    effective_config: dict[str, Any]
    mode: str
    supported_dimensions: dict[str, tuple[Any, ...]]
    seed: int
    status: str
    started_at: str
    finished_at: str
    cases: list[MatrixCaseResult]
    artifact_paths: dict[str, str]
    exit_code: int


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]}"


def validate_matrix(config: MatrixConfig) -> None:
    """Validate dimensions and combinations before any case can execute."""
    for name, values in config.dimensions.items():
        allowed = SUPPORTED_DIMENSIONS.get(name)
        if allowed is None:
            message = f"unsupported dimension: {name}"
            raise MatrixValidationError(message)
        if not values:
            message = f"dimension {name} must have at least one value"
            raise MatrixValidationError(message)
        unsupported = [value for value in values if value not in allowed]
        if unsupported:
            if name == "rerank_enabled" and True in unsupported:
                raise MatrixValidationError("reranking is not supported by this installation")
            message = (
                f"value(s) {unsupported!r} for dimension {name!r} are not supported; "
                f"supported values: {list(allowed)!r}"
            )
            raise MatrixValidationError(message)
    for case in _raw_combinations(config.dimensions):
        if case.get("rerank_enabled") and case.get("provider") not in {None, "ollama"}:
            raise MatrixValidationError("reranking is not supported by this provider")


def _raw_combinations(dimensions: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = sorted(dimensions)
    values = [sorted(dimensions[name], key=_canonical) for name in names]
    combinations = itertools.product(*values)
    return [dict(zip(names, combination, strict=True)) for combination in combinations] or [{}]


def expand_matrix(config: MatrixConfig) -> list[ExpandedCase]:
    """Return deterministic Cartesian cases with IDs independent of input order."""
    validate_matrix(config)
    return [
        ExpandedCase(case_id=_stable_id("case", case), effective_config=case)
        for case in _raw_combinations(config.dimensions)
    ]


def fixture_config() -> MatrixConfig:
    """Return the dependency-free profile used for contract smoke tests."""
    return MatrixConfig(
        matrix_id="fixture",
        profile="fixture",
        dataset=DatasetReference(
            dataset_id="localrag-core", dataset_version="1.0.0", split="smoke", checksum="fixture"
        ),
        dimensions={"retrieval_mode": ["hybrid", "vector"]},
    )


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def _git_provenance() -> tuple[str, bool]:
    root = Path(__file__).parent.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("LOCALRAG_CODE_REVISION", "unknown"), True
    return revision, dirty


def run_matrix(
    config: MatrixConfig,
    output_dir: Path,
    *,
    executor: Callable[[ExpandedCase, Path], dict[str, Any]] | None = None,
    dry_run: bool = False,
    tracker: TrackingSession | None = None,
) -> dict[str, Any]:
    """Execute cases in isolated directories and write the canonical manifest."""
    cases = expand_matrix(config)
    matrix_dir = output_dir / config.matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    code_revision, code_dirty = _git_provenance()
    manifest: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": config.matrix_id,
        "run_id": _stable_id("run", {"matrix_id": config.matrix_id, "seed": config.seed}),
        "profile": config.profile,
        "dataset": config.dataset.model_dump(),
        "corpus": {"identity": config.dataset.dataset_id, "checksum": config.dataset.checksum},
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "model": {"provider": "ollama", "embedding": "nomic-embed-text", "generation": "gemma3:4b"},
        "effective_config": config.model_dump(),
        "mode": config.mode,
        "supported_dimensions": SUPPORTED_DIMENSIONS,
        "seed": config.seed,
        "status": "dry_run" if dry_run else "running",
        "started_at": started,
        "cases": [],
        "artifact_paths": {"matrix": str(matrix_dir)},
    }
    tracking = tracker or TrackingSession()
    tracking.start_parent(
        manifest["run_id"],
        {
            "matrix_id": config.matrix_id,
            "profile": config.profile,
            "dataset": config.dataset.model_dump(),
        },
    )
    for case in cases:
        case_dir = matrix_dir / "cases" / case.case_id
        record: dict[str, Any] = {
            "case_id": case.case_id,
            "run_id": _stable_id("run", {"matrix_id": config.matrix_id, "case_id": case.case_id}),
            "matrix_id": config.matrix_id,
            "dataset": config.dataset.model_dump(),
            "corpus": {"identity": config.dataset.dataset_id, "checksum": config.dataset.checksum},
            "code_revision": code_revision,
            "code_dirty": code_dirty,
            "model": {
                "provider": "ollama",
                "embedding": "nomic-embed-text",
                "generation": "gemma3:4b",
            },
            "supported_dimensions": SUPPORTED_DIMENSIONS,
            "seed": config.seed,
            "effective_config": case.effective_config,
            "mode": config.mode,
            "status": "planned" if dry_run else "running",
            "started_at": datetime.now(UTC),
            "metrics": {},
            "latency": {"value": 0.0, "unit": "seconds"},
            "resources": {"unit": "unknown"},
            "artifact_paths": {"work": str(case_dir)},
        }
        tracking.start_case(case.case_id, {"case_id": case.case_id, **case.effective_config})
        if not dry_run:
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                with span(SpanName.BENCHMARK, {"run_id": record["run_id"], "stage": "case"}):
                    result = executor(case, case_dir) if executor else {"metrics": {}}
                record.update(result)
                record["status"] = result.get("status", "complete")
            except Exception as exc:  # failure is a result, not a runner abort
                record["status"] = "failed"
                record["error"] = _error(exc)
        tracking.log_metrics(record.get("metrics", {}))
        tracking.log_artifacts(
            [
                Path(path)
                for path in record.get("artifact_paths", {}).values()
                if Path(path).is_file()
            ]
        )
        tracking.finish_case(record["status"], record.get("error", {}).get("message"))
        record["finished_at"] = datetime.now(UTC)
        manifest["cases"].append(record)
    manifest["status"] = (
        "dry_run"
        if dry_run
        else (
            "failed"
            if any(case["status"] == "failed" for case in manifest["cases"])
            else "complete"
        )
    )
    manifest["finished_at"] = datetime.now(UTC)
    manifest["exit_code"] = 0 if manifest["status"] in {"dry_run", "complete"} else 1
    path = matrix_dir / "manifest.json"
    path.write_text(json.dumps(manifest, default=str, indent=2, sort_keys=True), encoding="utf-8")
    validated = MatrixManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    tracking.log_artifacts([path])
    tracking.finish_parent(manifest["status"], None)
    tracking.close()
    return validated.model_dump(mode="json")
