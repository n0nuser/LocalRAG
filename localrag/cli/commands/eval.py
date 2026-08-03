"""CLI command to run the RAGAS evaluation suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

_RUNNER = Path(__file__).parent.parent.parent.parent / "evals" / "run_evals.py"


def eval_suite(
    api_url: str = typer.Option("http://localhost:8000", help="LocalRAG API base URL."),
    api_key: str = typer.Option("", help="X-API-Key header value (empty = no auth)."),
    offline: bool = typer.Option(
        default=False,
        help="Skip live API calls; use stored contexts from the dataset.",
    ),
    seed: int = typer.Option(42, help="Seed for down-sampling and judge sampling."),
    sample: int = typer.Option(
        0, help="Evaluate only N examples (0 = all), chosen deterministically from the seed."
    ),
    dataset: str = typer.Option("localrag-core", help="Registered dataset_id."),
    version: str = typer.Option("", help="Dataset version (empty = highest registered)."),
    split: str = typer.Option("default", help="Named split to evaluate."),
) -> None:
    """Run the RAGAS evaluation suite and print a pass/fail summary."""
    cmd = [
        sys.executable,
        str(_RUNNER),
        f"--api-url={api_url}",
        f"--seed={seed}",
        f"--dataset={dataset}",
        f"--split={split}",
    ]
    if api_key:
        cmd.append(f"--api-key={api_key}")
    if offline:
        cmd.append("--offline")
    if sample > 0:
        cmd.append(f"--sample={sample}")
    if version:
        cmd.append(f"--version={version}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    raise typer.Exit(result.returncode)
