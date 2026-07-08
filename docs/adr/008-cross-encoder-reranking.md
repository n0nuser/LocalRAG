# ADR 008: Optional cross-encoder reranking as a final relevance step

## Context

Hybrid vector+BM25 retrieval (ADR 005) with RRF fusion ranks well on lexical
and semantic signal separately, but neither embedding similarity nor BM25
score directly optimizes for "is this chunk actually relevant to this exact
question" the way a cross-encoder (which scores the pair jointly) does.
Running a cross-encoder over the full corpus per query would be too slow;
running it only on the fused top-k throws away recall it could have fixed.

## Decision

Make reranking an optional, off-by-default final step:

- When `RERANK_ENABLED=true`, `Retriever.retrieve` over-fetches
  `RERANK_FETCH_K` candidates (instead of the usual `top_k * 2`) from the
  vector/hybrid path.
- `CrossEncoderReranker` (`localrag/rag/reranker.py`), backed by a local
  `sentence-transformers` cross-encoder model (`RERANK_MODEL`, default
  `cross-encoder/ms-marco-MiniLM-L-6-v2`), scores each `(question, chunk)`
  pair and trims back down to `top_k`.
- Reranking runs on the raw candidate list, strictly before freshness decay
  (ADR 006) and parent-section expansion — it is the last relevance-ordering
  step; freshness/expansion apply to the reranked, already-trimmed result.
- `sentence-transformers` is an optional dependency (`uv sync --extra
  rerank`); nothing imports it unless the feature is enabled, keeping the
  default install lightweight.

## Consequences

- Retrieval quality can improve for ambiguous queries without changing the
  default (disabled) behavior or the default dependency footprint.
- Extra latency and a model download when enabled — appropriate to enable
  per-deployment, not universally.
- `_fuse_results`'s dead RRF weight branch (only ever taken when
  `bm25_weight == 0.5`, mathematically identical to the other branch) was
  removed as part of this change since it was mechanically adjacent cleanup,
  not a new decision in itself.

## Related

`[[005-hybrid-retrieval]]`, `[[006-freshness-decay]]`,
`docs/superpowers/plans/2026-07-07-production-rag-hardening.md` (Task 6).
