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

`retrieval_recall` is the recall counterpart: the fraction of
annotation-relevant citations that retrieval actually surfaced. Threshold 0.8.
`citation_accuracy` scores precision over what the *answer* cited, so neither it
nor the LLM-judged `context_recall` could catch retrieval quietly returning
topically similar passages instead of the ones that answer the question — this
metric is identity-level and needs no judge.

It joins the two sides two ways, because the run modes name chunks differently:

| Mode | Retrieved IDs are | Join |
| --- | --- | --- |
| Offline | Dataset citation IDs (`offline_retrieved_citation_ids`) | Exact, on ID |
| Live | Corpus chunk hashes | Citation text against retrieved context text |

The namespace is proven by overlap with the record's declared citation IDs, not
assumed. The text join accepts normalized containment or 60% token coverage
(`RETRIEVAL_RECALL_TOKEN_COVERAGE`), because chunk boundaries cut passages and
an annotated citation is usually a subset of a larger retrieved chunk. When
neither join is possible the case is `unavailable` — never zero, which would be
indistinguishable from retrieval genuinely finding nothing.

The `context_omission` failure label uses the same join
(`evals.metrics.resolve_retrieved_citations`), so the metric and the label can
never disagree. Before that was shared, the label was joined on IDs alone and
was wrong in both modes: offline the retrieved IDs *were* the citation list, so
the difference was always empty and the label could never fire; live they were
corpus hashes sharing no namespace with citation IDs, so it fired for every
record.

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
