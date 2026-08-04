# ADR 026: Long-context benchmark boundary

## Context

The existing `eval --offline` path evaluates stored answers and contexts. It is
useful as a reproducible fixture control, but it does not exercise retrieval,
prompt construction, or a live answering model. Long-context comparisons also
cannot assume that a local model supports a particular window.

## Decision

The first live-local slice is a manually invoked profile in the canonical matrix
runner. It uses the versioned dataset citation corpus, stable score/citation-ID
ordering, deduplication, and either fixed top-five retrieval or deterministic
stuffing. Input budget is the requested native window less fixed prompt overhead
and reserved output tokens. The whitespace token estimator is named in each
configuration; an oversized first chunk fails rather than being clipped.

Before a case runs, Ollama `/api/show` is probed for the model digest and native
context length. Windows above that limit are `unsupported` with a reason; a
missing probe is `unavailable`. Neither state receives a quality score. The
initial matrix is `gemma3:4b` at 4096, 8192, and 32768 requested tokens. This is
an artifact/configuration matrix, not a claim that all three windows work.

## Measurement

Each live case records mode, dataset/corpus identity, model digest, seed,
effective context construction and token counts, EM/F1, retrieval/generation/
scoring/total latency, and explicit resource measurement statuses. CPU RSS and
GPU VRAM are unavailable in this first slice because no portable sampler is
part of the benchmark boundary; warm/cold state is recorded as unknown. Model
request errors and timeouts are failed cases. The current fixture is not a
long-context distractor or multi-hop dataset, so its results are a control and
not evidence for broad retrieval claims.

## Consequences

The benchmark remains local and manually invoked, with no automatic RAGAS
workflow. Capability gaps are visible in manifests instead of being confused
with model failures or scores. A later slice can add annotated distractor data,
model-native tokenizers, and scoped resource samplers without changing these
semantics.
