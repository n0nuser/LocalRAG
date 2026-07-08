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

## Cross-encoder reranking (optional)

Disabled by default (`RERANK_ENABLED=false`). When enabled, `Retriever.retrieve`
(`localrag/rag/retriever.py`) over-fetches `RERANK_FETCH_K` candidates from the
vector/hybrid path instead of the default `top_k * 2`, and a local
`cross-encoder/ms-marco-MiniLM-L-6-v2` model (`RERANK_MODEL`, served via
`sentence-transformers` — install with `uv sync --extra rerank`) re-scores each
`(question, chunk_text)` pair through `CrossEncoderReranker.rerank`
(`localrag/rag/reranker.py`) and trims the candidate list down to `top_k`,
best-first, adding a `rerank_score` key to each context.

Reranking runs on the raw fused/vector candidate list — **before**
`apply_freshness` and `_expand_to_parent_section` — so it acts as the final
relevance step, and freshness decay / parent-section expansion still apply to
the reranked, already-trimmed top-`k` results (matching how those two behave
when reranking is disabled). `localrag/api/dependencies.py::get_reranker`
builds the `CrossEncoderReranker` only when `RERANK_ENABLED=true`, mirroring
the pluggable-provider shape in `localrag/llm/factory.py` (nothing imports
`sentence-transformers` unless the feature is turned on).

## Low-confidence refusal gate

`RAGEngine.stream_chat_from_contexts` (`localrag/rag/engine.py`) can short-circuit
before calling the LLM at all: if the top retrieved context's `score` is below
`RAG_MIN_CONTEXT_SCORE`, or no contexts were retrieved while the gate is enabled,
`_is_low_confidence` returns `True` and `_low_confidence_response` yields a single
canned refusal token plus a `final` event with `sources: []` and
`low_confidence: True`. Otherwise generation proceeds as usual and the `final`
event carries `low_confidence: False`. The flag is surfaced end-to-end:
`QueryResponse.low_confidence` in `POST /query`, and in the `final` SSE payload
for `POST /query/stream`.

Disabled by default (`RAG_MIN_CONTEXT_SCORE=0.0`). Because the score scale
depends on the embedding model and `RETRIEVAL_MODE` (raw cosine/L2 similarity
vs. fused hybrid/RRF scores), there is no universal threshold — tune it
per-corpus after inspecting typical top scores for known-good and known-bad
queries. This is a lightweight heuristic gate, not a replacement for a real
guardrails layer (e.g. NeMo Guardrails) in a regulated setting.

## Query rewriting (optional)

Disabled by default (`QUERY_REWRITE_ENABLED=false`). When enabled,
`Retriever.retrieve` (`localrag/rag/retriever.py`) calls
`rewrite_query(question, settings)` (`localrag/rag/query_rewrite.py`) before
embedding/BM25 search. `rewrite_query` reuses
`localrag/llm/factory.py::build_provider` (the same `ResilientProvider`-wrapped
provider abstraction used for answering) with a retrieval-specific system
prompt that asks for a short, keyword-dense reformulation of the question,
preserving exact identifiers/codes/names verbatim.

The rewrite is retrieval-only: it replaces the text sent to
`OllamaEmbedder.embed_text` and `Bm25Index.query`, but the original question is
still what gets passed to the reranker (if enabled) and to the final answer
prompt — rewriting never affects citations or the text the LLM sees when
generating the answer. On any provider failure (timeout, exception, empty
response) `rewrite_query` logs and falls back to the original question, so
retrieval degrades to its normal behavior rather than failing the request.

This adds one extra LLM round-trip per query, so it is off by default; enable
it when lexical/embedding mismatch between conversational questions and
indexed document phrasing is hurting retrieval recall.

## Tenant tagging (optional)

Per Chroma's own multi-tenancy guidance, this project uses a `tenant_id`
metadata field filtered at query time (via the metadata pre-filtering above)
rather than per-tenant Chroma collections, which the Chroma Cookbook
explicitly warns fragments the HNSW index and breaks whole-collection
operations like `Bm25Index.from_vector_store`.

`TENANT_ID` (`localrag/settings.py`, empty by default) is written to every
chunk's metadata at ingest time (`localrag/ingestion/service.py::_ingest_one`).
A caller scopes retrieval to one tenant by passing
`{"question": "...", "metadata_filter": {"tenant_id": "team-a"}}` to
`POST /query` — no new retrieval code is needed since this reuses the
`metadata_filter` mechanism described above.

This is an equality-filter convenience for a small-team shared deployment, not
a security boundary — anyone with API access can still query across all
`tenant_id` values by omitting the filter; pair with `API_KEY` and, if genuine
per-tenant isolation is ever required, revisit as a dedicated
(out-of-scope-for-this-plan) access-control project.

## Query caching (optional)

Disabled by default (`QUERY_CACHE_TTL_SECONDS=0`). When enabled (set a positive
TTL in seconds), `POST /query` (`localrag/api/routers/query.py`) is served
through an in-process `QueryCache` (`localrag/rag/query_cache.py`), wired via
`localrag/api/dependencies.py::get_query_cache` (an `lru_cache`-memoized
singleton, so all requests within one process share the same cache) and
passed into `localrag.api.service.query_json`.

Cache keys are an exact-match SHA-256 hash over the normalized question
(stripped, lowercased), `model`, `n_results`, and `retrieval_mode`
(`make_cache_key`) — this is not semantic/fuzzy matching, so any change to
wording, model, result count, or retrieval mode is a cache miss. Cached values
are the full serialized `QueryResponse` (`response.model_dump()`), so a cache
hit replays `answer`, `sources` (including `heading_path`/`chunk_type`),
`latency_ms`, `model`, and `low_confidence` exactly as they were on the
original request — a cached low-confidence refusal is served back as
low-confidence, not silently upgraded.

`QUERY_CACHE_MAXSIZE` bounds the number of entries (least-recently-used
eviction via `cachetools.TTLCache`) independent of the TTL.

This cache is in-process only — it is **not** shared across multiple
`uvicorn` worker processes (each worker gets its own `QueryCache` instance),
so cache hit rate degrades as worker count increases. If LocalRAG ever runs
multi-process/multi-replica in production, a shared external cache (e.g.
Redis) would be the upgrade path; that is out of scope for the current
single-process deployment model.

Cache hits are not separately audit-logged — a served-from-cache response
still only produces the `query_cache_hit` log line, since it bypasses the
retriever and LLM call entirely.

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
- `tenant_id`

The retriever returns `freshness_factor` and `ingested_at` in contexts so rank
decisions are visible in API and test traces.

`content_hash` also drives incremental rebuild — `POST /collections/rebuild` skips
re-embedding any source whose file bytes haven't changed.
