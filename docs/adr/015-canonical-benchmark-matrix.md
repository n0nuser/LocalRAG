# ADR 015: Canonical benchmark matrix runner

## Context

Comparing configurations through separate scripts produces unstable case names,
shared state, and incompatible reports. A benchmark needs one expansion and
artifact contract.

## Decision

`evals/matrix.py` owns the canonical matrix contract and orchestration. Versioned
dimensions and manifests are validated against supported capabilities, expanded
as a deterministic Cartesian product, and assigned IDs from canonicalized
configuration values. Each case receives an isolated work directory and a
structured result or failure; independent cases continue after a failure.

The CLI exposes built-in `fixture` and `embedding-comparison` profiles as well
as versioned JSON matrix configuration. The embedding profile is a named matrix
profile, not a separate benchmark implementation. Reports and leaderboards are
consumers of matrix/result artifacts, never alternate runners.

## Consequences

- Reordered dimension input cannot silently rename or reorder cases.
- Failures remain inspectable without discarding successful cases.
- Adding a supported dimension requires updating the versioned capability
  contract and its consumers rather than adding an ad hoc script.
- Current profiles and supported values deliberately describe the local Ollama
  capability envelope, not every possible provider or model.

## Related

[Reproducibility](../reproducibility.md), [Evaluation reports](../evaluation-reports.md), [ADR 016](016-bounded-parallel-evaluation.md)
