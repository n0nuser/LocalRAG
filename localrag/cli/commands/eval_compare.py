"""CLI adapter for versioned evaluation result comparison."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

_COMPARER = Path(__file__).parent.parent.parent.parent / "evals" / "compare.py"


def eval_compare(
    candidate: Path = typer.Argument(..., exists=True, readable=True),
    baseline: Path | None = typer.Option(None, readable=True, help="Explicit baseline JSON."),
    baseline_name: str | None = typer.Option(None, help="Named file under evals/baselines/."),
    threshold: list[str] | None = typer.Option(None, help="Gate, e.g. faithfulness>=0.60."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable output."),  # noqa: FBT003
) -> None:
    """Compare a result against an explicit file or named reviewed baseline."""
    if (baseline is None) == (baseline_name is None):
        raise typer.BadParameter("provide exactly one of --baseline or --baseline-name")
    cmd = [sys.executable, str(_COMPARER), str(candidate)]
    cmd += ["--baseline", str(baseline)] if baseline else ["--baseline-name", baseline_name or ""]
    for expression in threshold or []:
        cmd += ["--threshold", expression]
    if json_output:
        cmd.append("--json")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    raise typer.Exit(result.returncode)
