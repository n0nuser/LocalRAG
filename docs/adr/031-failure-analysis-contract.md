# ADR 031: Per-case failure analysis contract

## Status

Accepted

## Decision

Failure analysis is an additive section of the canonical #84 `ResultFile`. It
consumes the #82 dataset judgments, #73 runner artifacts, and #74 per-case
metric outcomes after those inputs have been assembled. It never creates a
second run identity or metric schema.

The stable labels are `retrieval_miss`, `context_omission`,
`unsupported_claim`, `wrong_citation`, `out_of_scope_refusal`,
`evaluator_failure`, and `unclassified`. A case has an ordered primary label
and may have secondary labels. Counts are per case per label; `failed_count` is
deduplicated by case. Deterministic evidence is ordered before an explicit,
optional local judge. A judge must return structured allowed labels, confidence,
and model identity; timeout, retry exhaustion, and invalid output never invent
a label.

Examples are exact: no retrieved IDs/text is `retrieval_miss`; retrieved
context missing an annotated relevant citation is `context_omission`; a failed
faithfulness or hallucination threshold is `unsupported_claim`; an answer
citation outside retrieved IDs is `wrong_citation`; explicit phrases such as
"I don't know" or "outside my scope" are `out_of_scope_refusal`; execution or
metric status `error` is `evaluator_failure`. A retrieval miss suppresses the
derived omission label, and evaluator failure suppresses unsupported-claim
inference from the failed judge. Reliable independent evidence may still
produce multiple labels, ordered by the taxonomy above. Empty answers and
malformed or missing annotations use `unclassified` with a reason.

Cases fail when execution fails, an evaluator errors, or a metric crosses its
declared threshold. Unavailable and non-finite values remain unavailable and
cannot be treated as zero. Missing citation annotations therefore produce
`unavailable` metric status and `unclassified` analysis. Reports contain IDs,
labels, confidence, reasons, and counts only. Questions, answers, contexts,
documents, and source paths are not exported by default.

## Consequences

The analysis is auditable and deterministic for reliable evidence, while
heuristic labels remain interpretations rather than ground truth. Ambiguous
cases can be judged locally only when explicitly enabled. RAGAS and the
manual-only evaluation workflow remain unchanged.
