"""CLI adapter for the canonical benchmark matrix runner."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evals.dataset.checksum import manifest_checksum
from evals.dataset.errors import DatasetError
from evals.dataset.registry import load_dataset
from evals.long_context import make_live_executor
from evals.matrix import (
    DatasetReference,
    MatrixConfig,
    MatrixValidationError,
    fixture_config,
    run_matrix,
)


def _profile(name: str, dataset_id: str) -> MatrixConfig:
    if name == "fixture":
        fixture = fixture_config()
        return fixture.model_copy(
            update={"dataset": fixture.dataset.model_copy(update={"dataset_id": dataset_id})}
        )
    if name == "embedding-comparison":
        dataset = load_dataset(dataset_id)
        return MatrixConfig(
            matrix_id="embedding-comparison",
            profile=name,
            dataset=DatasetReference(
                dataset_id=dataset_id,
                dataset_version=dataset.dataset_version,
                split="smoke",
                checksum=manifest_checksum(dataset),
            ),
            dimensions={"embedding_model": ["nomic-embed-text"]},
        )
    if name == "hyde":
        dataset = load_dataset(dataset_id)
        return MatrixConfig(
            matrix_id="hyde-retrieval",
            profile=name,
            dataset=DatasetReference(
                dataset_id=dataset_id,
                dataset_version=dataset.dataset_version,
                split="smoke",
                checksum=manifest_checksum(dataset),
            ),
            dimensions={
                "provider": ["ollama"],
                "retrieval_mode": ["hybrid"],
                "retrieval_experiment_mode": ["baseline", "rewrite", "hyde", "rewrite+hyde"],
            },
        )
    if name == "long-context":
        dataset = load_dataset(dataset_id)
        return MatrixConfig(
            matrix_id="long-context",
            profile=name,
            dataset=DatasetReference(
                dataset_id=dataset_id,
                dataset_version=dataset.dataset_version,
                split="smoke",
                checksum=manifest_checksum(dataset),
            ),
            dimensions={
                "generation_model": ["gemma3:4b"],
                "context_window": [4096, 8192, 32768],
                "context_strategy": ["fixed_top_k", "stuff"],
                "top_k": [5],
            },
            mode="live-local",
        )
    message = f"unknown profile {name!r}; use fixture, embedding-comparison, hyde, or long-context"
    raise typer.BadParameter(message)


def benchmark(  # noqa: C901
    dataset: str = typer.Option(
        "localrag-core", help="Registered dataset ID for a built-in profile."
    ),
    config: Path | None = typer.Option(None, "--config", help="JSON matrix configuration."),
    matrix: Path | None = typer.Option(
        None, "--matrix", help="Alias for --config: JSON matrix configuration."
    ),
    profile: str = typer.Option("fixture", help="Built-in matrix profile."),
    result_output: Path = typer.Option(
        Path("evals/results/matrices"),
        "--result-output",
        "--output-dir",
        help="Matrix artifact root.",
    ),
    seed: int | None = typer.Option(None, min=0, help="Override the matrix seed."),
    offline: bool = typer.Option(
        True,  # noqa: FBT003
        help="Run in offline mode; never fall back to remote APIs.",
    ),
    mode: str | None = typer.Option(
        None, help="Execution semantics: fixture-offline (stored artifacts) or live-local (Ollama)."
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434", help="Ollama URL for live-local benchmark profiles."
    ),
    dry_run: bool = typer.Option(
        default=False, help="Expand and validate without executing cases."
    ),
) -> None:
    """Run a manually-invoked benchmark matrix."""
    if config and matrix:
        raise typer.BadParameter("use only one of --config and --matrix")
    try:
        config_path = config or matrix
        if config_path:
            matrix_config = MatrixConfig.model_validate(
                json.loads(config_path.read_text(encoding="utf-8"))
            )
        else:
            matrix_config = _profile(profile, dataset)
        updates: dict[str, object] = {}
        if seed is not None:
            updates["seed"] = seed
        if mode is not None:
            updates["mode"] = mode
        elif offline:
            updates["mode"] = "fixture-offline"
        if updates:
            matrix_config = matrix_config.model_copy(update=updates)
        executor = None
        if matrix_config.mode == "live-local" and not dry_run:
            records = load_dataset(
                matrix_config.dataset.dataset_id, matrix_config.dataset.dataset_version
            ).split(matrix_config.dataset.split)
            model = str(matrix_config.dimensions.get("generation_model", ["gemma3:4b"])[0])
            executor = make_live_executor(
                records,
                base_url=ollama_url,
                model=model,
                seed=matrix_config.seed,
                timeout=120.0,
            )
        if executor is None:
            manifest = run_matrix(matrix_config, result_output, dry_run=dry_run)
        else:
            manifest = run_matrix(matrix_config, result_output, dry_run=dry_run, executor=executor)
    except (DatasetError, OSError, json.JSONDecodeError, MatrixValidationError, ValueError) as exc:
        typer.echo(f"benchmark configuration failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    required_keys = {"status", "cases", "artifact_paths", "exit_code"}
    if (
        not isinstance(manifest, dict)
        or not required_keys.issubset(manifest)
        or not isinstance(manifest["exit_code"], int)
        or not isinstance(manifest["artifact_paths"], dict)
    ):
        typer.echo("benchmark returned a malformed canonical result", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))
    raise typer.Exit(manifest["exit_code"])
