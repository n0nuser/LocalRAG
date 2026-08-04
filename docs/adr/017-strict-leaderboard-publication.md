# ADR 017: Strict leaderboard publication

## Context

A leaderboard is a public claim, not an evaluation execution environment.
Guessing missing provenance or silently combining incompatible measurements turns
an attractive table into an unreliable benchmark.

## Decision

`evals/leaderboard.py` is a strict publication adapter over reviewed canonical
artifacts. It validates schema, dataset identity, evaluation contract, exact
model and embedding identities, retrieval/reranker/chunker configuration,
hardware, temperature, repetitions, metric definitions, measurement units,
code, dependency, and seed provenance. It rejects missing, malformed,
duplicate, non-finite, or incompatible rows and can require an exact model
identity matrix.

Publication never runs a benchmark, samples data, fills missing measurements, or
infers a result. Rows use immutable model revisions and digests, document the
comparability envelope and variance (including repetitions and cold/warm state),
and are ordered deterministically. Markdown and optional JSON are projections of
the validated publication contract.

## Consequences

- An empty or rejected publication is preferable to an unsupported ranking.
- Producers must capture provenance before review; the adapter never guesses it.
- Hardware, runtime, seed, dataset, configuration, and temperature differences
  remain visible comparability boundaries.
- Leaderboard output is deterministic for unchanged inputs and is separate from
  benchmark execution and experiment tracking.

## Related

[Benchmark leaderboard](../benchmark-leaderboard.md), [ADR 013](013-versioned-benchmark-results.md), [ADR 015](015-canonical-benchmark-matrix.md)
