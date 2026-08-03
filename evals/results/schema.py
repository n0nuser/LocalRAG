"""Canonical, versioned evaluation result contract.

Version zero is the result shape emitted before #84. It is migrated explicitly
to version one; future versions are rejected rather than guessed at.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CURRENT_SCHEMA_VERSION = 1
MetricDirection = Literal["higher_is_better", "lower_is_better"]
MissingPolicy = Literal["missing", "not_applicable"]
ResultStatus = Literal["complete", "partial", "failed"]


class ResultError(ValueError):
    """A result cannot be loaded, migrated, or validated."""


class DatasetIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_version: str
    split: str
    checksum: str


class MetricDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    direction: MetricDirection
    threshold: float | None = None
    unit: str | None = None
    missing_value: MissingPolicy = "missing"


class MetricCaseResult(BaseModel):
    """Schema-compatible per-record value and evaluation outcome."""

    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    threshold: float | None = None
    status: Literal["complete", "unavailable", "error"] = "complete"
    input_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    warning: str | None = None


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    descriptor: MetricDescriptor
    value: float | None = None
    cases: dict[str, float | None] = Field(default_factory=dict)
    non_finite_cases: list[str] = Field(default_factory=list)
    case_results: dict[str, MetricCaseResult] = Field(default_factory=dict)
    valid_count: int = 0
    missing_count: int = 0
    error_count: int = 0

    @field_validator("value", mode="before")
    @classmethod
    def _finite_or_missing(cls, value: Any) -> float | None:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None


class ResultFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = CURRENT_SCHEMA_VERSION
    run_id: str
    timestamp: datetime
    dataset: DatasetIdentity
    selected_ids: list[str]
    metrics: list[MetricResult]
    provenance: dict[str, Any] = Field(default_factory=dict)
    status: ResultStatus = "complete"

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value > CURRENT_SCHEMA_VERSION:
            message = (
                f"result schema_version {value} is newer than supported version "
                f"{CURRENT_SCHEMA_VERSION}; upgrade LocalRAG"
            )
            raise ResultError(message)
        if value != CURRENT_SCHEMA_VERSION:
            message = f"result schema_version {value} must be migrated before validation"
            raise ResultError(message)
        return value

    @field_validator("metrics")
    @classmethod
    def _unique_metrics(cls, value: list[MetricResult]) -> list[MetricResult]:
        names = [metric.descriptor.name for metric in value]
        if len(names) != len(set(names)):
            raise ValueError("metrics must have unique names")
        return value

    def metric_map(self) -> dict[str, MetricResult]:
        return {metric.descriptor.name: metric for metric in self.metrics}

    def model_dump_json_safe(self) -> str:
        return self.model_dump_json(indent=2)


def _migrate_v0(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate the pre-#84 runner shape without inventing missing values."""
    dataset = raw.get("dataset")
    environment = raw.get("environment", {})
    if not isinstance(dataset, dict):
        raise ResultError("historical result is missing the dataset object")
    scores = raw.get("scores")
    if not isinstance(scores, dict):
        raise ResultError("historical result is missing scores")
    metrics = [
        {
            "descriptor": {
                "name": name,
                "direction": "higher_is_better",
                "missing_value": "missing",
            },
            "value": value,
        }
        for name, value in scores.items()
    ]
    timestamp = raw.get("timestamp")
    if not isinstance(timestamp, str):
        raise ResultError("historical result is missing timestamp")
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "run_id": timestamp,
        "timestamp": timestamp,
        "dataset": {
            "dataset_id": dataset.get("dataset_id"),
            "dataset_version": dataset.get("dataset_version"),
            "split": dataset.get("split"),
            "checksum": dataset.get("checksum"),
        },
        "selected_ids": dataset.get("selected_record_ids", []),
        "metrics": metrics,
        "provenance": environment,
        "status": "complete",
    }


def migrate_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ResultError("result must be a JSON object")
    version = raw.get("schema_version", 0)
    if not isinstance(version, int):
        raise ResultError("schema_version must be an integer")
    if version == 0:
        return _migrate_v0(raw)
    if version > CURRENT_SCHEMA_VERSION:
        message = (
            f"result schema_version {version} is newer than supported version "
            f"{CURRENT_SCHEMA_VERSION}; upgrade LocalRAG"
        )
        raise ResultError(message)
    if version != CURRENT_SCHEMA_VERSION:
        message = f"no migration is registered for result schema_version {version}"
        raise ResultError(message)
    return raw


def load_result(path: Path) -> ResultFile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"could not read result {path}: {exc}"
        raise ResultError(message) from exc
    try:
        return ResultFile.model_validate(migrate_result(raw))
    except ResultError:
        raise
    except Exception as exc:
        message = f"result {path} failed validation: {exc}"
        raise ResultError(message) from exc
