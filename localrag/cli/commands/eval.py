"""CLI command to run the RAGAS evaluation suite."""

from __future__ import annotations

import subprocess
import sys

import typer

# Invoked as a module, not a file path: `evals` ships in the wheel, so resolving
# it relative to __file__ would break for an installed package.
_RUNNER_MODULE = "evals.run_evals"


def eval_suite(
    api_url: str = typer.Option("http://localhost:8000", help="LocalRAG API base URL."),
    api_key: str = typer.Option("", help="X-API-Key header value (empty = no auth)."),
    offline: bool = typer.Option(
        default=False,
        help="Skip live API calls; use stored contexts from the dataset.",
    ),
    seed: int | None = typer.Option(
        None,
        help=(
            "Seed for down-sampling and judge sampling. "
            "Unset uses EVAL_SEED env var, then a built-in default (42)."
        ),
    ),
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
        "-m",
        _RUNNER_MODULE,
        f"--api-url={api_url}",
        f"--dataset={dataset}",
        f"--split={split}",
    ]
    if api_key:
        cmd.append(f"--api-key={api_key}")
    if offline:
        cmd.append("--offline")
    if seed is not None:
        cmd.append(f"--seed={seed}")
    if sample > 0:
        cmd.append(f"--sample={sample}")
    if version:
        cmd.append(f"--version={version}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    raise typer.Exit(result.returncode)
