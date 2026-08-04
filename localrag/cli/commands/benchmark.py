"""CLI adapter for the canonical benchmark matrix runner."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evals.dataset.checksum import manifest_checksum
from evals.dataset.errors import DatasetError
from evals.dataset.registry import load_dataset
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
    message = f"unknown profile {name!r}; use fixture or embedding-comparison"
    raise typer.BadParameter(message)


def benchmark(
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
        if offline:
            updates["mode"] = "offline"
        if updates:
            matrix_config = matrix_config.model_copy(update=updates)
        manifest = run_matrix(matrix_config, result_output, dry_run=dry_run)
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
