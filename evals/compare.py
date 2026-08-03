"""Command-line benchmark result comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.results.compare import ThresholdError, compare, parse_threshold
from evals.results.schema import ResultError, load_result

BASELINES_DIR = Path(__file__).parent / "baselines"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a benchmark result with an explicit baseline."
    )
    parser.add_argument("candidate", type=Path, help="Canonical or historical result JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", type=Path)
    group.add_argument("--baseline-name")
    parser.add_argument("--threshold", action="append", default=[])
    parser.add_argument("--json", action="store_true", dest="machine_output")
    args = parser.parse_args(argv)
    baseline_path = args.baseline or BASELINES_DIR / f"{args.baseline_name}.json"
    try:
        baseline, candidate = load_result(baseline_path), load_result(args.candidate)
        thresholds = [
            parse_threshold(value, set(candidate.metric_map())) for value in args.threshold
        ]
        report = compare(baseline, candidate, thresholds)
    except (OSError, ResultError, ThresholdError, ValueError) as exc:
        sys.stderr.write(f"compare error: {exc}\n")
        return 2
    if args.machine_output:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(("COMPARABLE" if report.comparable else "NOT COMPARABLE") + "\n")
        for delta in report.deltas:
            sys.stdout.write(
                f"{delta.name}: absolute={delta.absolute} relative={delta.relative}"
                + (" REGRESSION" if delta.regression else "")
                + "\n"
            )
        for issue in report.regressions + report.incompatibilities:
            sys.stdout.write(f"- {issue}\n")
    if not report.comparable:
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
