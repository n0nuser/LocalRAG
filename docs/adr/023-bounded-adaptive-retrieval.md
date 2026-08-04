# ADR 023: Bounded adaptive retrieval

## Status

Accepted

## Decision

LocalRAG uses an opt-in, bounded evidence policy rather than an autonomous
agent loop. The controller follows `INITIAL_RETRIEVE -> EVALUATE_EVIDENCE`,
then either answers, escalates `k`, performs at most the configured number of
retrieval-only refinements, or abstains. Every execution terminates at
`ANSWER` or `ABSTAIN` and records a final `DONE` transition.

Evidence confidence is a corpus-tuned heuristic policy, not calibrated model
self-confidence. It combines non-empty evidence, top score, score margin,
source diversity, and lexical query coverage. Structured provider output may
refine one retrieval query, but provider prose, hidden chain-of-thought, and
raw self-confidence are never trace evidence. HyDE is unsupported until a
separate explicit policy is accepted.

The original user query remains the answer-generation query and is retained in
the trace. Existing rewrite, query expansion, reranking, metadata filters,
freshness, parent expansion, and compression remain inside the retriever's
existing order. Refinement changes retrieval only. Stable `source#chunk_index`
IDs deduplicate evidence across rounds.

Hard settings cap rounds, refinements, `k`, wall time, and provider work. Empty
or metadata-filtered results, repeated evidence, invalid refinement, provider
failure, timeout, and budget exhaustion abstain deterministically while
retaining best evidence only for observability. Context overflow is represented
as a stop reason for future providers and must not trigger retries.

The additive trace contains policy/version, typed state transitions, round,
query kind, requested/returned `k`, stable hit IDs, measurable evidence
signals, decision, provider/model, latency, token/cost estimates, and stop
reason. It never contains hidden reasoning. JSON and SSE share the same engine
path, and disabled mode preserves the prior single-shot behavior.

Issue #72's unbounded iterative-agent proposal is consolidated here and is not
a separate dependency. Evaluation remains the existing deterministic,
RAGAS, and manual-only workflow.

## Consequences

The defaults keep the feature disabled. Operators must tune thresholds against
known good and bad queries for their corpus and should compare the bounded mode
with the fixed single-shot baseline using the existing benchmark tooling.
