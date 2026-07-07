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

1. Vector search (`VectorStore.query`) ranked by embedding distance. The Chroma
   collection is created with `hnsw:space=cosine` (`localrag/storage/vector_store.py`),
   matching the cosine-similarity objective most embedding models (including the
   default `nomic-embed-text`) are trained against. Chroma applies collection
   metadata only at creation time — an existing collection created before this
   setting keeps its original (`l2`) space; delete and re-ingest (or
   `POST /collections/rebuild` after a manual `delete_collection`) to pick up
   `cosine`.
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

## Chunk overlap

`CHUNK_OVERLAP_CHARS` (default 150, ~12.5% of `CHUNK_MAX_CHARS=1200`) only applies
where `localrag/ingestion/structural_chunker.py::_split_long_paragraph` must
hard-split a single paragraph that exceeds `chunk_max_chars`. Adjacent
*packed* structural chunks (the common case — `_pack_blocks`) are
deliberately disjoint with zero overlap: boundary-awareness (never splitting
mid-table, mid-code-block, or mid-heading-section) substitutes for overlap
there. This is an intentional design choice, not an oversight.

## Metadata pre-filtering

`POST /query`'s `QueryRequest.metadata_filter` (`localrag/api/schemas.py`) accepts
an optional equality-only `dict[str, str]` filter applied to chunk metadata
**before** ranking, e.g.:

```json
{"question": "...", "metadata_filter": {"source": "/docs/handbook.pdf"}}
```

It threads through `RAGEngine.answer` / `RAGEngine.stream_answer`
(`localrag/rag/engine.py`) into `Retriever.retrieve`'s `metadata_filter`
parameter (`localrag/rag/retriever.py`), which applies it on **both** retrieval
paths:

1. **Vector search** — passed natively as Chroma's `where=` clause via
   `VectorStore.query(embedding, top_k, where=...)`
   (`localrag/storage/vector_store.py`), so Chroma itself excludes
   non-matching chunks before the HNSW search returns results.
2. **BM25 search** (hybrid mode only) — applied client-side as an equality
   check against each BM25 hit's metadata via the `_matches_filter` helper in
   `localrag/rag/retriever.py`, since `rank_bm25` has no native filter concept.

This is **equality-only** — it is not a full Chroma `$and`/`$or`/`$in` query
DSL. Every key/value pair in `metadata_filter` must match exactly
(`metadata.get(key) == value`) for a chunk to survive filtering; there is no
support for ranges, negation, or boolean combinators. Pairs naturally with the
`source`/`file_type` fields already written on every chunk during ingestion.

## Parent-section expansion

After ranking, fusion, and freshness decay, `Retriever._expand_to_parent_section`
(`localrag/rag/retriever.py`) expands top retrieval hits that carry a non-empty
`heading_path` chunk metadata value to the **full sibling-chunk section** they
belong to, via `VectorStore.get_chunks_by_heading(source, heading_path)`
(`localrag/storage/vector_store.py`), which fetches every chunk sharing that
`source` + `heading_path` pair and returns them sorted by `chunk_index`. The
merged section text is joined with `"\n\n"` and stored on the context dict as
`expanded_text`, while the originally matched chunk's `text` (and
`chunk_index`) are retained unchanged so citations (`SourceRef`) still point at
the precise matched chunk. `localrag/rag/prompt.py::build_prompt` prefers
`expanded_text` over `text` when present when composing the LLM prompt, so the
model sees the whole section instead of just the one matching sentence.

Controlled by `PARENT_EXPANSION_ENABLED` (default `true`); set to `false` to
skip expansion and prompt with only the originally matched chunk text. Hits
with an empty `heading_path`, or whose section has only a single chunk, are
left unexpanded.

## Ingestion metadata dependencies

Freshness and debugging depend on chunk metadata written during ingestion:

- `source`
- `chunk_index`
- `ingested_at`
- `heading_path`
- `chunk_type`
- `content_hash`
- `source_mtime`
- `git_commit`

The retriever returns `freshness_factor` and `ingested_at` in contexts so rank
decisions are visible in API and test traces.

`content_hash` also drives incremental rebuild — `POST /collections/rebuild` skips
re-embedding any source whose file bytes haven't changed.
