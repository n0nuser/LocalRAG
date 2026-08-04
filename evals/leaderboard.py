"""Validate canonical benchmark publications and render a deterministic leaderboard."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LEADERBOARD_SCHEMA_VERSION = 1


class LeaderboardError(ValueError):
    """A source artifact cannot support a published comparison."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dataset(_Strict):
    id: str
    version: str
    split: str
    hash: str = Field(min_length=1)


class ModelIdentity(_Strict):
    provider: str
    name: str
    revision: str
    digest: str = Field(min_length=1)
    quantization: str
    runtime: str


class EmbeddingIdentity(_Strict):
    name: str
    revision: str
    digest: str = Field(min_length=1)


class Retrieval(_Strict):
    mode: str
    config: dict[str, Any]


class Reranker(_Strict):
    id: str
    version: str
    enabled: bool


class Chunker(_Strict):
    id: str
    version: str
    config: dict[str, Any]


class Hardware(_Strict):
    os: str
    cpu: str
    gpu: str
    ram_gb: float = Field(gt=0)


class Summary(_Strict):
    definition: str
    unit: str
    mean: float
    median: float
    p95: float
    stddev: float

    @field_validator("mean", "median", "p95", "stddev")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("summary values must be finite")
        return value

    @field_validator("stddev")
    @classmethod
    def non_negative_stddev(cls, value: float) -> float:
        if value < 0:
            raise ValueError("stddev must be non-negative")
        return value


class LatencySummary(_Strict):
    unit: str
    method: str
    mean: float
    median: float
    p95: float
    stddev: float

    @field_validator("mean", "median", "p95", "stddev")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency values must be finite")
        return value

    @field_validator("stddev")
    @classmethod
    def non_negative_stddev(cls, value: float) -> float:
        if value < 0:
            raise ValueError("stddev must be non-negative")
        return value

    @field_validator("mean", "median", "p95")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("latency values must be non-negative")
        return value


class ResourceSummary(_Strict):
    unit: str
    method: str
    peak: float = Field(ge=0)


class Run(_Strict):
    timestamp: datetime
    code_revision: str
    dependency_lock_hash: str
    seed: int = Field(ge=0)


class LeaderboardArtifact(_Strict):
    schema_version: int
    source_kind: Literal["matrix_case", "evaluation_result"]
    artifact_id: str
    source_artifact: str
    dataset: Dataset
    evaluation_schema_version: int
    model: ModelIdentity
    embedding: EmbeddingIdentity
    retrieval: Retrieval
    reranker: Reranker
    chunker: Chunker
    hardware: Hardware
    temperature: Literal["cold", "warm"]
    repetitions: int = Field(ge=3)
    warmup_repetitions: int = Field(ge=1)
    metrics: dict[str, Summary] = Field(min_length=1)
    latency: LatencySummary
    resources: ResourceSummary
    run: Run

    @field_validator("schema_version", "evaluation_schema_version")
    @classmethod
    def supported_version(cls, value: int) -> int:
        if value != LEADERBOARD_SCHEMA_VERSION:
            message = f"schema version {value} is not supported"
            raise ValueError(message)
        return value


class ExactIdentity(_Strict):
    provider: str
    name: str
    revision: str
    digest: str


class IdentityMatrix(_Strict):
    schema_version: int
    identities: list[ExactIdentity] = Field(min_length=1)


class LeaderboardResult(_Strict):
    schema_version: int
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class PublicationResult:
    """Files written by a successful publication."""

    output: Path
    row_count: int
    json_output: Path | None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"{path}: malformed JSON: {exc}"
        raise LeaderboardError(message) from exc


def _artifact(path: Path) -> LeaderboardArtifact:
    try:
        artifact = LeaderboardArtifact.model_validate(_load_json(path))
    except Exception as exc:
        message = f"{path}: invalid leaderboard artifact: {exc}"
        raise LeaderboardError(message) from exc
    return artifact


def _identity(row: LeaderboardArtifact) -> tuple[str, str, str, str]:
    return (row.model.provider, row.model.name, row.model.revision, row.model.digest)


def _compatibility(row: LeaderboardArtifact) -> tuple[Any, ...]:
    return (
        row.dataset.model_dump(mode="json"),
        row.evaluation_schema_version,
        row.embedding.model_dump(mode="json"),
        row.retrieval.model_dump(mode="json"),
        row.reranker.model_dump(mode="json"),
        row.chunker.model_dump(mode="json"),
        row.temperature,
        sorted((name, metric.definition, metric.unit) for name, metric in row.metrics.items()),
        row.latency.unit,
        row.resources.unit,
    )


def _row(row: LeaderboardArtifact) -> dict[str, Any]:
    return {
        "source_kind": row.source_kind,
        "artifact_id": row.artifact_id,
        "source_artifact": row.source_artifact,
        "dataset": row.dataset.model_dump(mode="json"),
        "evaluation_schema_version": row.evaluation_schema_version,
        "model": row.model.model_dump(mode="json"),
        "embedding": row.embedding.model_dump(mode="json"),
        "retrieval": row.retrieval.model_dump(mode="json"),
        "reranker": row.reranker.model_dump(mode="json"),
        "chunker": row.chunker.model_dump(mode="json"),
        "hardware": row.hardware.model_dump(mode="json"),
        "temperature": row.temperature,
        "repetitions": row.repetitions,
        "warmup_repetitions": row.warmup_repetitions,
        "metrics": {
            name: row.metrics[name].model_dump(mode="json") for name in sorted(row.metrics)
        },
        "latency": row.latency.model_dump(mode="json"),
        "resources": row.resources.model_dump(mode="json"),
        "run": row.run.model_dump(mode="json"),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LocalRAG Benchmark Leaderboard",
        "",
        "Generated only from validated canonical benchmark artifacts. "
        "Empty means no measured results have been published.",
        "",
        "| Model identity | Dataset | Temperature | Repetitions | Quality metrics | "
        "Latency | Peak resource | Run |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        model = row["model"]
        dataset = row["dataset"]
        metrics = "; ".join(
            f"{name} mean={value['mean']} median={value['median']} "
            f"p95={value['p95']} sd={value['stddev']} {value['unit']}"
            for name, value in row["metrics"].items()
        )
        latency = row["latency"]
        resources = row["resources"]
        lines.append(
            f"| `{model['provider']}/{model['name']}@{model['revision']} ({model['digest']})` | "
            f"`{dataset['id']}@{dataset['version']}:{dataset['split']}` (`{dataset['hash']}`) | "
            f"{row['temperature']} | {row['repetitions']} "
            f"(+{row['warmup_repetitions']} warm-up) | {metrics} | "
            f"mean={latency['mean']} p95={latency['p95']} {latency['unit']} "
            f"({latency['method']}) | "
            f"{resources['peak']} {resources['unit']} ({resources['method']}) | "
            f"`{row['run']['timestamp']}` `{row['run']['code_revision']}` |"
        )
    if not rows:
        lines.extend(["| No valid benchmark artifacts | | | | | | | |"])
    return "\n".join(lines) + "\n"


def generate_leaderboard(
    paths: list[Path], output: Path, json_output: Path | None = None, *, matrix: Path | None = None
) -> PublicationResult:
    """Validate source artifacts and write deterministic Markdown/JSON output."""
    rows = [_artifact(path) for path in paths]
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise LeaderboardError("duplicate model/config identity in source artifacts")
    if rows and any(_compatibility(row) != _compatibility(rows[0]) for row in rows[1:]):
        raise LeaderboardError(
            "incompatible dataset, configuration, temperature, or metric definitions"
        )
    if matrix:
        try:
            expected = IdentityMatrix.model_validate(_load_json(matrix))
        except Exception as exc:
            message = f"{matrix}: invalid exact identity matrix: {exc}"
            raise LeaderboardError(message) from exc
        expected_keys = {
            (item.provider, item.name, item.revision, item.digest) for item in expected.identities
        }
        missing = expected_keys - set(identities)
        if missing:
            message = f"missing exact model identities: {sorted(missing)}"
            raise LeaderboardError(message)
    rendered = [_row(row) for row in sorted(rows, key=lambda item: _identity(item))]
    payload = LeaderboardResult(schema_version=LEADERBOARD_SCHEMA_VERSION, rows=rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(rendered), encoding="utf-8")
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return PublicationResult(output, len(rendered), json_output)
