"""Container entrypoint for the canonical LocalRAG benchmark contracts.

The fixture command is deliberately offline and delegates case identity and
matrix manifests to ``evals.matrix``. It only adapts deterministic fixture
metrics into the versioned result schema; it does not select or compare data
itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evals.dataset.checksum import manifest_checksum
from evals.dataset.registry import load_dataset
from evals.matrix import DatasetReference, MatrixConfig, run_matrix
from evals.metrics import exact_match, f1
from evals.results.schema import (
    EvaluationCaseResult,
    MetricDescriptor,
    MetricResult,
    ResultFile,
    load_result,
)

ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = ROOT / "evals" / "results" / "docker"
MODEL_LOCK = ROOT / "docker" / "models.lock.json"


class BenchmarkConfigError(ValueError):
    """The container benchmark configuration is unsafe or incomplete."""


class ModelPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    digest: str


class ModelLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    models: list[ModelPin] = Field(min_length=1)


def load_model_lock(path: Path = MODEL_LOCK) -> ModelLock:
    try:
        return ModelLock.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkConfigError(f"invalid model lock {path}: {exc}") from exc  # noqa: EM102


def validate_model_lock(lock: ModelLock, required: set[str]) -> dict[str, str]:
    pins = {model.name: model.digest for model in lock.models}
    if len(pins) != len(lock.models):
        raise BenchmarkConfigError("model lock contains duplicate model names")
    missing = sorted(required - pins.keys())
    if missing:
        raise BenchmarkConfigError(f"model lock missing {', '.join(missing)}")  # noqa: EM102
    for name, digest in pins.items():
        if not digest.startswith("sha256:") or len(digest.removeprefix("sha256:")) != 64:
            message = f"model {name!r} must use an immutable digest"
            raise BenchmarkConfigError(message)
    return pins


def reset_results(output: Path) -> None:
    """Make repeated runs idempotent and prevent stale exports being consumed."""
    if output.exists():
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)


def _uv_path() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise BenchmarkConfigError("uv is required for live-local execution")
    return uv


def _request_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(f"service is not ready at {url}: {exc}") from exc  # noqa: EM102
    if not isinstance(payload, dict):
        message = f"service returned a non-object response at {url}"
        raise BenchmarkConfigError(message)
    return payload


def wait_for_http(url: str, *, timeout: float = 60.0, interval: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "unknown error"
    while time.monotonic() < deadline:
        try:
            _request_json(url, timeout=min(interval, 5.0))
            return  # noqa: TRY300
        except BenchmarkConfigError as exc:
            last_error = str(exc)
            time.sleep(interval)
    raise BenchmarkConfigError(f"timed out waiting for {url}: {last_error}")  # noqa: EM102


def verify_ollama_models(url: str, required: set[str], lock: ModelLock) -> None:
    pins = validate_model_lock(lock, required)
    models = _request_json(f"{url.rstrip('/')}/api/tags").get("models", [])
    actual = {
        item.get("name"): item.get("digest")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for name in sorted(required):
        actual_name = name if name in actual else f"{name}:latest"
        if actual_name not in actual:
            message = f"required model {name!r} is missing from Ollama"
            raise BenchmarkConfigError(message)
        if actual[actual_name] != pins[name]:
            raise BenchmarkConfigError(
                f"model {name!r} digest mismatch: expected {pins[name]}, got {actual[actual_name]}"  # noqa: EM102
            )


def _fixture_result(output: Path, *, profile: str, seed: int) -> ResultFile:
    dataset = load_dataset("localrag-core", "1.0.0")
    records = dataset.split("smoke")
    dataset_ref = DatasetReference(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        split="smoke",
        checksum=manifest_checksum(dataset),
    )
    matrix = MatrixConfig(
        matrix_id="docker-fixture",
        profile=profile,
        dataset=dataset_ref,
        dimensions={"retrieval_mode": ["hybrid", "vector"]},
        seed=seed,
        mode="fixture-offline",
    )

    def execute(_case: Any, _work: Path) -> dict[str, Any]:
        answers = [record.reference_answer for record in records]
        return {
            "metrics": {
                "exact_match": sum(exact_match(answer, [answer]) for answer in answers)
                / len(answers),
                "f1": sum(f1(answer, [answer]) for answer in answers) / len(answers),
            }
        }

    manifest = run_matrix(matrix, output / "matrices", executor=execute)
    timestamp = datetime.now(UTC)
    case_results = [
        EvaluationCaseResult(
            record_id=record.record_id,
            stage="retrieval",
            status="completed",
        )
        for record in records
    ]
    metrics = [
        MetricResult(
            descriptor=MetricDescriptor(name=name, direction="higher_is_better", threshold=1.0),
            value=1.0,
            cases={record.record_id: 1.0 for record in records},
            case_results={
                record.record_id: {"value": 1.0, "status": "complete"} for record in records
            },
            valid_count=len(records),
        )
        for name in ("exact_match", "f1")
    ]
    result = ResultFile(
        run_id=manifest["run_id"],
        timestamp=timestamp,
        dataset=dataset_ref.model_dump(),
        selected_ids=[record.record_id for record in records],
        metrics=metrics,
        provenance={
            "evaluation_mode": "fixture-offline",
            "container_profile": profile,
            "seed": seed,
            "matrix_manifest": "matrices/docker-fixture/manifest.json",
            "reproducibility_claim": (
                "identical fixture metadata and comparator decisions; no model claim"
            ),
        },
        cases=case_results,
    )
    result_path = output / "result.json"
    result_path.write_text(result.model_dump_json_safe(), encoding="utf-8")
    return load_result(result_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or prepare a container benchmark.")
    parser.add_argument("command", choices=("run", "prepare"))
    parser.add_argument(
        "--mode", choices=("fixture-offline", "live-local"), default="fixture-offline"
    )
    parser.add_argument("--profile", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--api-url", default="http://localrag-api:8000")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            wait_for_http(f"{args.ollama_url.rstrip('/')}/api/tags")
            verify_ollama_models(
                args.ollama_url,
                {"gemma3:4b", "nomic-embed-text"},
                load_model_lock(),
            )
            return 0
        reset_results(args.output)
        if args.mode == "live-local":
            wait_for_http(f"{args.api_url.rstrip('/')}/health")
            verify_ollama_models(
                args.ollama_url,
                {"gemma3:4b", "nomic-embed-text"},
                load_model_lock(),
            )
            completed = subprocess.run(  # noqa: S603
                [
                    _uv_path(),
                    "run",
                    "localrag",
                    "benchmark",
                    "--profile",
                    "long-context",
                    "--mode",
                    "live-local",
                    "--result-output",
                    str(args.output / "matrices"),
                    "--ollama-url",
                    args.ollama_url,
                ],
                cwd=ROOT,
                check=False,
            )
            return completed.returncode
        result = _fixture_result(args.output, profile=args.profile, seed=42)
        return result.exit_code  # noqa: TRY300
    except BenchmarkConfigError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
