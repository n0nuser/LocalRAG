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
) -> None:
    """Run the RAGAS evaluation suite and print a pass/fail summary."""
    cmd = [sys.executable, str(_RUNNER), f"--api-url={api_url}"]
    if api_key:
        cmd.append(f"--api-key={api_key}")
    if offline:
        cmd.append("--offline")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    raise typer.Exit(result.returncode)
