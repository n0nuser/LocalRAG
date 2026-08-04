# ADR 013: Versioned benchmark result documents

## Context

Evaluation output needs to survive schema evolution, preserve per-case failure
information, and support safe comparisons. Reports, baselines, and publication
must not each invent a subtly different result shape.

## Decision

Use the versioned `ResultFile` JSON contract as the canonical evaluation result.
It contains schema/run/timestamp identity, dataset identity and selected IDs,
metric descriptors and values, per-case values and statuses, counts,
provenance, overall status, failures, and exit code. Metric descriptors declare
direction, threshold, unit, and missing-value policy. Non-finite values remain
missing and are never converted to zero.

Loaders migrate the pre-contract version 0 shape through an explicit migration.
Unsupported future versions and unregistered historical versions fail instead
of being guessed at. Comparisons are direction-aware, report metric/case
additions and removals, and reject incompatible dataset or provenance inputs.
Only an explicitly selected reviewed baseline is used.

Exit codes distinguish pass (`0`), comparable regression or failed threshold
(`1`), and usage, schema, missing, or incompatible input (`2`). Reports,
comparison tooling, and leaderboard publication consume this canonical schema;
they do not redefine it.

## Consequences

- Historical artifacts remain readable only through named, reviewable migrations.
- Per-case and missing/error data prevent aggregate scores from hiding failures.
- Baselines are deliberate review artifacts rather than implicit latest results.
- Every consumer inherits one compatibility and provenance contract.

## Related

[Reproducibility](../reproducibility.md), [Evaluation reports](../evaluation-reports.md), [ADR 017](017-strict-leaderboard-publication.md)
