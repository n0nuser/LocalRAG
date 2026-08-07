# ADR 041: Claim scope-applicability filtering

## Status

Accepted

## Context

Retrieval selects passages by topical similarity, which is blind to the
qualifier that decides whether a passage answers the question. A question about
a single occurrence and a passage about habitual exposure measured over years
are the same topic, so both rank and both reach the model.

The pipeline was `retrieve -> concatenate -> generate`. Nothing between
retrieval and generation asked whether a retrieved claim *applies* at the scope
the question asks about. The observed failure (#172, #173, #174) was a grounded
answer that restated long-term epidemiological findings as the consequence of a
single event, with `low_confidence` false throughout: every stage did its job,
and no stage owned the question.

The default system prompt now instructs the model to preserve claim scope
(#173). That is the cheap mitigation and it covers the common case, but it
relies on the answering model honoring an instruction while it is also
composing an answer — which small local models do unevenly.

## Decision

LocalRAG adds an **opt-in, bounded applicability filter** between retrieval and
generation, `localrag/rag/claim_filter.py`.

The stage makes **exactly one** provider call regardless of context count, and
asks a single closed question: which of these numbered passages do not apply at
the question's scope. The provider replies with a minimal JSON object naming
passage indices. The stage's only power is **removal of a context that was
already retrieved** — it never rewrites a passage, never adds one, never
reorders, and never contributes to the answer text.

Scope usually lives in the heading rather than the sentence, so the filter's
prompt includes each passage's `heading_path` (ADR 004, and #172). Judging
applicability without it is the blind case that produced the original failure.

Filtering runs **before** compression, so the compression budget is spent only
on passages that can answer the question.

Sources reported to the caller come from the **filtered** set. A discarded
passage did not inform the answer, so citing it would misattribute the
response.

### Failure is always degradation, never refusal

Every failure path returns the **unfiltered** contexts: provider error,
unparseable output, and out-of-range indices. Answering with more context than
strictly necessary is the behavior that shipped before this stage existed;
answering with too little because one judgment call went wrong is a regression.

A verdict that discards *every* context is refused for the same reason. The
engine already has an abstain path for genuinely insufficient evidence
(`_is_low_confidence`, `RAG_MIN_CONTEXT_SCORE`); a filter that empties the
context set is far more likely to be a bad verdict than a correct one.

Prompt construction deliberately sits **outside** the degradation guard: a
failure there is a bug in this module, not an unreachable provider, and
silently degrading would hide it.

### Provider support

Unlike HyDE (ADR 025), this stage is **not** Ollama-only. It calls
`generate_from_prompt`, which is on the `BaseLLMProvider` contract, so it works
identically on the Ollama, OpenAI, and Anthropic backends with no per-backend
guard.

### Observability

The stage's observation is merged into the existing query `trace` field
alongside the HyDE trace: status, evaluated and discarded counts, latency,
model, and the discarded `source#chunk_index` IDs. When the stage is disabled
it contributes nothing, so an otherwise-unmodified query still reports the same
trace as before.

## Consequences

The default keeps the feature **disabled**, consistent with every other
optional stage (rerank, HyDE, query rewrite, expansion, adaptive). Enabling it
costs one extra provider call per query, which is significant on local
hardware.

Its value is a hypothesis until measured. No current metric expresses "the
model answered a different question than the one asked", and the eval harness
has no scope regression case yet (#174). Operators enabling this should compare
against the disabled baseline on their own corpus using the existing benchmark
tooling rather than assuming an improvement.

The stage is deliberately narrow: it filters retrieved contexts by scope
applicability only. Per-claim scope markers in the answer, and claim extraction
as a distinct representation, are explicitly **not** part of this decision.
