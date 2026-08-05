# Evaluation Metrics

> Part of the [evaluation documentation](evaluation.md) — start there for the pipeline overview.

Metric implementation semantics are pinned by `evals/metrics.py` and the
RAGAS version in `pyproject.toml`/`uv.lock`. A result records
`metric_contract_version`, the RAGAS prompt/version, judge seed, endpoint, and
model digests in provenance.

See [ADR 014](adr/014-evaluation-metric-contract.md) for the durable metric decision.

## Deterministic metrics

`exact_match` normalizes Unicode with NFKC, applies Unicode case-folding,
replaces punctuation with spaces, collapses whitespace, and compares the
complete normalized strings. With multiple references, the maximum match is
used. Both empty answers match; an empty answer against a non-empty reference
does not.

`f1` uses the same normalization and whitespace tokenization. It computes
multiset overlap, `precision = overlap / predicted_tokens`,
`recall = overlap / reference_tokens`, and
`F1 = 2 * precision * recall / (precision + recall)`. The maximum score across
references is used. Two empty token lists score 1; only one empty list scores
0. These metrics make no model or network calls.

## Judge and citation metrics

The existing RAGAS metrics remain enabled: faithfulness, answer relevancy,
context precision, and context recall. `hallucination_rate` is the pinned
answer-level complement `1 - faithfulness`; lower is better. Judge exceptions
and non-finite responses are recorded per case as `error`, not converted to 0.
The local Ollama judge uses temperature 0 and the run seed.

`citation_accuracy` is annotation-backed: cited stable `citation_id` values
must be present and are scored as citation-ID precision against the relevant
IDs from the #82 dataset judgments. Missing or malformed citation annotations
are `unavailable`/missing, never zero or perfect. Citation IDs are scoped to a
record and validated against its declared citations before evaluation.

## Results and thresholds

Every metric stores an aggregate value, direction, threshold, per-case value,
status, input IDs, warnings/errors, and valid/missing/error counts. Aggregates
average valid cases only. Missing values remain `null`; a metric with no valid
cases is not passing. Thresholds are inclusive: higher-is-better metrics pass
at `value >= threshold`, while lower-is-better metrics pass at
`value <= threshold`.

Failure analysis is a consumer of these per-case artifacts, not another metric.
It classifies failed cases as `retrieval_miss`, `context_omission`,
`unsupported_claim`, `wrong_citation`, `out_of_scope_refusal`,
`evaluator_failure`, or `unclassified`. Labels are ordered primary-plus-
secondary labels, and counts are per label while failed-case totals are
deduplicated. Missing citation annotations remain unavailable and classify as
unclassified; no citation score is invented. See [ADR 031](adr/031-failure-analysis-contract.md).

The canonical JSON contract and migrations are in `evals/results/schema.py`.

## Parallel evaluation

The execution policy is [ADR 016](adr/016-bounded-parallel-evaluation.md).

The runner preserves the selected record order and writes case records in that
order even when completion order differs. Live retrieval/generation uses the
lowest of the retrieval, generation, and total limits; the default is 1
generation request and 2 in-flight rows. Judge calls default to 1, embeddings
to 1, metric rows to 4, and total work to 4. Override these with
`--*-concurrency` flags. These defaults are intentionally conservative for a
single local Ollama instance; increase them only after checking GPU memory and
provider queueing.

`--case-timeout` defaults to 120 seconds. A timeout cancels the row and its
child provider work. Cancelling the runner cancels and awaits all child tasks.
Provider, timeout, and metric failures are recorded per row in `cases`, with
`failure_counts`; independent rows continue. A run exits 0 only when all rows
complete and all metric thresholds pass, and exits 1 otherwise. Invalid CLI
configuration exits 2.

Concurrency does not promise bit-for-bit judge equality. Compare EM/F1 exactly
for deterministic fixtures; compare judge-backed metrics with documented
tolerance (the existing guidance treats deltas below roughly 0.02 as noise).

## Live context boundary

Live evaluation obtains answer text from `POST /query` and retrieved chunk text
and stable `chunk_id` values from the authenticated benchmark-only
`POST /query/contexts` endpoint. It never substitutes source paths or fixture
contexts for a live response. The endpoint is deliberately separate from the
normal public query response because raw document text may contain private data;
benchmark clients must be explicitly authenticated and handle the returned text
as sensitive evaluation input.
