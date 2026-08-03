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


def _profile(name: str) -> MatrixConfig:
    if name == "fixture":
        return fixture_config()
    if name == "embedding-comparison":
        dataset = load_dataset("localrag-core")
        return MatrixConfig(
            matrix_id="embedding-comparison",
            profile=name,
            dataset=DatasetReference(
                dataset_id="localrag-core",
                dataset_version=dataset.dataset_version,
                split="smoke",
                checksum=manifest_checksum(dataset),
            ),
            dimensions={"embedding_model": ["nomic-embed-text"]},
        )
    message = f"unknown profile {name!r}; use fixture or embedding-comparison"
    raise typer.BadParameter(message)


def benchmark(
    matrix: Path | None = typer.Option(None, help="JSON matrix configuration."),
    profile: str = typer.Option("fixture", help="Built-in matrix profile."),
    output_dir: Path = typer.Option(Path("evals/results/matrices"), help="Matrix artifact root."),
    dry_run: bool = typer.Option(
        default=False, help="Expand and validate without executing cases."
    ),
) -> None:
    """Run a manually-invoked benchmark matrix."""
    try:
        if matrix:
            config = MatrixConfig.model_validate(json.loads(matrix.read_text(encoding="utf-8")))
        else:
            config = _profile(profile)
        manifest = run_matrix(config, output_dir, dry_run=dry_run)
    except (DatasetError, OSError, json.JSONDecodeError, MatrixValidationError, ValueError) as exc:
        typer.echo(f"benchmark configuration failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))
    raise typer.Exit(manifest["exit_code"])
