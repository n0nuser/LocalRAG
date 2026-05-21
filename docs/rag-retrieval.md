# RAG retrieval design

This document describes how LocalRAG ranks context chunks before prompting the
LLM. The main principle is: retrieval quality dominates answer quality.

## Retrieval modes

Configuration lives in `localrag/settings.py` and `.env.example`.

- `RETRIEVAL_MODE=vector`: embedding-only retrieval from Chroma.
- `RETRIEVAL_MODE=hybrid` (default): embedding retrieval + BM25 retrieval, fused
  with reciprocal rank fusion (RRF).

## Hybrid ranking

`localrag/rag/retriever.py` combines candidates from:

1. Vector search (`VectorStore.query`) ranked by embedding distance.
2. BM25 lexical search (`Bm25Index.query`) ranked by lexical relevance.

Candidates are merged by reciprocal rank fusion:

`fused_score(d) = sum(1 / (rrf_k + rank_i(d)))`

Where `rank_i(d)` is rank position in vector or BM25 list. This avoids brittle
score normalization across different ranker scales.

## Freshness decay

After retrieval/fusion, LocalRAG applies an optional recency factor using each
chunk's `ingested_at` metadata:

`freshness_factor = 0.5 ** (age_days / freshness_half_life_days)`

Final score becomes:

`final_score = base_score * freshness_factor`

Set `FRESHNESS_HALF_LIFE_DAYS=0` to disable this behavior.

## Ingestion metadata dependencies

Freshness and debugging depend on chunk metadata written during ingestion:

- `source`
- `chunk_index`
- `ingested_at`
- `heading_path`
- `chunk_type`

The retriever returns `freshness_factor` and `ingested_at` in contexts so rank
decisions are visible in API and test traces.
