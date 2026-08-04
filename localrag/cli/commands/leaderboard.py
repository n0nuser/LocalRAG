"""CLI adapter for deterministic benchmark leaderboard publication."""

from __future__ import annotations

from pathlib import Path

import typer

from evals.leaderboard import LeaderboardError, generate_leaderboard


def leaderboard(
    inputs: list[Path] = typer.Argument(
        default_factory=list, help="Validated canonical publication artifacts."
    ),
    output: Path = typer.Option(Path("leaderboard.md"), "--output", "-o"),
    json_output: Path | None = typer.Option(None, "--json-output", help="Machine-readable output."),
    matrix: Path | None = typer.Option(None, "--matrix", help="Exact model identity matrix JSON."),
) -> None:
    """Publish a leaderboard without running benchmarks or inventing results."""
    try:
        result = generate_leaderboard(inputs, output, json_output, matrix=matrix)
    except LeaderboardError as exc:
        typer.echo(f"leaderboard generation failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"wrote {result.output} ({result.row_count} row(s))")
