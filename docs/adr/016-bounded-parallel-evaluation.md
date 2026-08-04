# ADR 016: Bounded parallel evaluation

## Context

Evaluation involves retrieval, generation, embeddings, judges, and metric
workloads with very different resource costs. Unbounded concurrency overloads a
local Ollama instance, while completion-order output makes runs hard to inspect.

## Decision

Use independent bounded concurrency limits for retrieval, generation, judge,
embedding, metric, and total work. Conservative defaults are chosen for a local
Ollama instance: retrieval 2, generation/judge/embeddings 1, metrics 4, total
4. Cases are returned and written in input order regardless of completion order.

Each case has a timeout; provider, metric, timeout, and other failures become
structured per-case outcomes and independent cases continue. Cancellation
cancels and awaits child tasks. The runner reports partial/failed work and uses
exit status rather than pretending an incomplete run passed.

Concurrency does not make LLM output bit-for-bit deterministic. Deterministic
EM/F1 can be compared exactly; judge-backed metrics use documented tolerances
(small deltas, roughly below 0.02 in current guidance, may be noise).

## Consequences

- Local defaults favor predictable resource use over maximum throughput.
- Operators can increase limits deliberately after checking GPU memory and
  provider queueing.
- Partial failures remain actionable and do not erase completed case results.
- Reproducibility claims cover ordering and inputs, not impossible LLM equality.

## Related

[Evaluation metrics](../evaluation-metrics.md), [Reproducibility](../reproducibility.md), [ADR 012](012-reproducible-evaluation-metadata.md)
