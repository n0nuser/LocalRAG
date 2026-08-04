# ADR 025: Bounded HyDE Retrieval Experiment

## Status

Accepted, experimental and disabled by default.

## Decision

HyDE is retrieval-only. The explicit arms are `baseline`, `rewrite`, `hyde`, and
`rewrite+hyde`; `auto` preserves the existing boolean settings. Composition is the
original question, optional existing rewrite, one bounded hypothetical-document
generation, then retrieval. Query expansion remains separate and is not silently
applied to generated text.

Only local Ollama may generate HyDE text. Other backends are reported as
`unsupported_provider` and retrieval falls back without a remote call. Provider
errors, timeouts, empty or malformed output, and embedding failures use the normal
non-HyDE retrieval path.

The hypothetical passage is used for dense embedding only. BM25 uses the original
question by default, or the rewritten query only with `HYDE_LEXICAL_INPUT=rewritten`.
Metadata filtering, RRF fusion, reranking, freshness, parent expansion, and
compression retain their existing order. The original question remains the answer
prompt and reranker query.

Input is capped at 2,000 characters and output at 4,000 characters / 512 whitespace
tokens, with a 30-second timeout. Temperature and seed inherit the LLM settings.
Raw hypothetical text is not logged or returned by default; metadata exposes only
mode, provider/model, latency, status, and fallback reason. `HYDE_LOG_CONTENT` is a
local debugging opt-in, not a benchmark default.

## Consequences

HyDE adds at most one local generation call and can increase latency. It must be
compared on the same #73 corpus snapshot and settings, reporting retrieval quality
separately from generation and retrieval latency. RAGAS and manual-only evaluation
remain unchanged; this ADR makes no quality claim.
