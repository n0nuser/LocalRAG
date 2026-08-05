# ADR 028: RAPTOR feasibility boundary

## Context

RAPTOR recursively clusters embedded chunks and summarizes clusters. It could
help broad or long-context questions, but adds LLM indexing cost, staleness,
storage, and a second retrieval space. LocalRAG currently has flat leaf chunks
in Chroma and must not silently mix summary vectors with that collection.

## Decision

Keep RAPTOR as a research-only, dependency-free prototype in
`research/raptor_spike/`. It is not wired into ingestion, API dependency
injection, Chroma, the default retriever, RAGAS, or manual evaluation paths.
The prototype uses deterministic seeded hash-partition clusters
(`cluster_count=2`, minimum size 2, reduction factor 2), truncating summaries to
800 characters and stopping at four levels or when another reduction is not
possible. Singleton and undersized/outlier clusters remain represented by
leaves and do not create a summary. A failed summary is skipped while children,
source chunk IDs, source IDs, and hashes remain intact.

Summary prompts must be versioned and the summarizer/model identity recorded.
The fixture uses `fake-summary-v1`; no quality claim is made for an LLM. Stable
IDs hash node kind, level, summary text, and sorted child IDs. Leaf content
hashes and embedding identity are persisted. The artifact is schema-versioned,
written with fsync plus atomic replace, and rejects incompatible schemas.
Changing leaves, embeddings, summarizer, prompt/config, or schema invalidates
the artifact. Updates/deletes rebuild from the retained leaf set; interrupted
writes leave the prior artifact untouched. This bounded recovery is preferable
to pretending partial summary reuse is safe.

Retrieval can explicitly select levels, applies level weights, deduplicates by
source, preserves citation serialization, and enforces a character context
budget. Flat leaf retrieval is the caller's explicit fallback. Freshness,
reranking, and compression are not silently applied to summary nodes; a future
integration must define their order and provenance-aware behavior.

## Consequences and gate

The prototype establishes contracts and edge-case behavior, not production
quality. Adopt only after a real local model run on a versioned #73/#84 dataset
shows at least +5 percentage points NDCG@10 over flat hybrid retrieval, warm
p95 no worse than 2x, peak memory no worse than 2x, and storage overhead no
worse than 5x unless the quality gain justifies it. It must also show bounded
staleness/rebuild cost, complete citations, offline-compatible dependencies,
and acceptable maintenance ownership. Missing evidence is a failure.

## Status

**Defer / reject default adoption.** The fixture is intentionally too small and
uses deterministic fake embeddings and summaries. Real LLM calls, token counts,
quality, cold-start latency, RSS/VRAM, multilingual behavior, prompt injection,
and production-scale Chroma compatibility are unsupported and must be measured
in a follow-up before reconsideration.
