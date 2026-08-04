# ADR 014: Evaluation metric contract

## Context

Metric names alone do not make scores comparable. Normalization, annotation
joins, judge behavior, thresholds, and missing data must be explicit and
preserved per record.

## Decision

Pin exact match and F1 to deterministic NFKC normalization, Unicode case-folding,
punctuation-to-space replacement, whitespace collapse, and whitespace tokens.
Multiple references use the best reference score; empty-input behavior is
defined by the implementation. These metrics make no model or network calls.

Citation accuracy is annotation-backed: answer citation IDs are joined to
record-scoped relevance judgments and scored as citation-ID precision. Missing
or malformed annotations are unavailable, never zero or perfect.

Faithfulness, answer relevancy, context precision, and context recall remain
enabled RAGAS metrics, judged through local Ollama. `hallucination_rate` is the
lower-is-better complement `1 - faithfulness`; it is not an independent judge.
Judge exceptions and non-finite values are per-case errors.

Every metric records direction, inclusive threshold, unit, valid/missing/error
counts, aggregate, and per-case results. Aggregates use valid cases only; no
valid cases is not passing, and missing/not-applicable values stay explicit.

## Consequences

- Deterministic metrics provide exact regression signals independent of a judge.
- RAGAS remains available for quality assessment but its model limits and
  variance are visible in provenance and must be interpreted with tolerance.
- Per-case preservation supports failure analysis instead of hiding exceptions
  in an average.

## Related

[Evaluation metrics](../evaluation-metrics.md), [ADR 009](009-offline-ragas-judge.md), [ADR 011](011-evaluation-dataset-contract.md)
