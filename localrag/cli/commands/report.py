"""CLI adapter for offline benchmark report generation."""

from __future__ import annotations

from pathlib import Path

import typer

from evals.report import generate_report


def report(
    inputs: list[Path] = typer.Argument(
        ..., help="Canonical result or matrix manifest JSON files."
    ),
    output: Path = typer.Option(
        Path("report.html"), "--output", "-o", help="HTML file to overwrite."
    ),
    strict: bool = typer.Option(False, help="Exit nonzero if any input cannot be loaded."),  # noqa: FBT003
) -> None:
    """Generate a self-contained local HTML report from benchmark artifacts."""
    result = generate_report(inputs, output)
    for error in result.errors:
        typer.echo(f"report input error: {error}", err=True)
    typer.echo(f"wrote {result.output} ({result.run_count} run(s))")
    if strict and result.errors:
        raise typer.Exit(1)
