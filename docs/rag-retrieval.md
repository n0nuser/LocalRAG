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

## Freshness / recency

How recency is applied depends on the retrieval mode, because the two modes
produce scores on very different scales.

**Hybrid mode** — recency joins the fusion as a third ranked list, alongside
vector and BM25:

`fused_score(d) = w_vec/(rrf_k + rank_vec) + w_bm25/(rrf_k + rank_bm25) + FRESHNESS_WEIGHT/(rrf_k + rank_recency)`

`FRESHNESS_WEIGHT` (default `0.15`) comes out of the relevance budget, so
`w_vec + w_bm25 = 1 - FRESHNESS_WEIGHT`. Relevance ties are broken by recency
before ranking, and chunks with no usable `ingested_at` take the middle recency
rank so missing metadata neither helps nor hurts.

This replaced a multiplicative decay that overwhelmed RRF's compressed score
range — see the amendment in [ADR 006](adr/006-freshness-decay.md) for the
measurements.

**Vector mode** — scores spread widely enough for the original multiplicative
decay to act as an intended tiebreaker, so it still applies:

`freshness_factor = 0.5 ** (age_days / freshness_half_life_days)`
`final_score = base_score * freshness_factor`

`freshness_factor` is reported in retrieval contexts in both modes, even where it
no longer moves the score.

Set `FRESHNESS_HALF_LIFE_DAYS=0` or `FRESHNESS_WEIGHT=0` to disable recency.

## Chunk overlap

Ingestion uses one `Chunk` contract for fixed, structural, and recursive
strategies. It records deterministic `chunk_id` metadata, source order,
provenance, and existing structural metadata. Offsets are intentionally absent
because current strategies normalize whitespace or repack blocks. Empty input
produces no chunks; oversized atomic input is retained with an `oversized`
marker. See [ADR 021](adr/021-chunking-strategy-contract.md). Recursive is the
first additional strategy; semantic, sentence-window, and parent-child modes
remain follow-up work. The latter must not be confused with the existing
retrieval-time parent-section expansion below.

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

## Query expansion (optional)

Expansion is a separate retrieval-stage operation, disabled by default
(`QUERY_EXPANSION_ENABLED=false`). When both transformations are enabled, the
fixed order is **original question -> one rewrite call -> one expansion call ->
per-variant retrieval -> cross-variant fusion -> reranking -> freshness ->
parent-section expansion**. Rewriting therefore never accidentally causes one
expansion call per generated query. With expansion disabled, rewrite-only
behavior remains unchanged.

The expansion provider must return `{"queries": ["..."]}` (a JSON list is also
accepted). The typed `QueryExpansionResult` records the original, optional
rewrite, accepted variants, rejected values/reasons, and fallback status.
Variants are stripped, deduplicated by case-folded whitespace-normalized text,
and reject blank, non-string, and overlong values. The original question is
always retained exactly when expansion is enabled, including exact identifiers
and codes; generated text is only a search query and is never treated as a
fact or answer source. Provider errors, timeouts, malformed output, and an
empty response fall back to the original plus the rewrite when they differ.

Fan-out is bounded to at most 8 variants, one expansion LLM call, and 100 total
candidate slots allocated across the vector/BM25 rank lists (the configured
limits are lower by default: 4 variants, 500 characters, and 40 candidates).
Vector and BM25
rank lists are passed explicitly to weighted RRF. A hit is identified by
`source + chunk_index`, duplicate hits retain their first provenance, and ties
are deterministic. Metadata filters apply to every variant's vector/BM25
search. Cross-variant fusion happens before the existing original-question
reranker, freshness handling, and parent-section expansion. Synonym maps remain
follow-up work; evaluation remains the existing RAGAS/manual-only workflow.

## HyDE experiment (optional)

HyDE is disabled by default (`HYDE_ENABLED=false`). Use the explicit
`RETRIEVAL_EXPERIMENT_MODE` arms `baseline`, `rewrite`, `hyde`, and `rewrite+hyde`
for comparisons; `auto` preserves the existing boolean settings. The `rewrite+hyde`
order is one rewrite call followed by one hypothetical-document call. The generated
passage is embedded for dense retrieval only. BM25 uses the original question by
default (`HYDE_LEXICAL_INPUT=original`) to avoid generated-term drift; selecting the
rewritten lexical input is an explicit measured experiment.

HyDE calls only Ollama, uses bounded prompt/output/token limits, and inherits the
configured temperature and seed. Unsupported providers, timeouts, provider errors,
empty/malformed output, and embedding errors fall back to the non-HyDE path. Typed
trace metadata reports the arm, provider/model, latency, status, and fallback reason;
raw hypothetical text is excluded by default. Retrieval then follows the existing
metadata filter, RRF fusion, reranking, freshness, parent expansion, and compression
order. The original question remains the answer and reranker input.

Run the small reproducible #73 smoke profile manually with a fixed seed:

```bash
uv run localrag benchmark --profile hyde --dataset localrag-core --seed 42
```

Report retrieval quality and generation/retrieval latency separately. This profile
makes no broad quality claim and does not alter the RAGAS/manual-only workflow.

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

## Bounded adaptive retrieval

`ADAPTIVE_ENABLED` is disabled by default. When enabled, the engine runs the
typed controller in `localrag/rag/adaptive.py`:

`INITIAL_RETRIEVE -> EVALUATE_EVIDENCE -> ANSWER | ESCALATE | REFINE -> RETRIEVE -> EVALUATE_EVIDENCE -> ANSWER | ABSTAIN -> DONE`.

The controller is bounded by `ADAPTIVE_MAX_ROUNDS`,
`ADAPTIVE_MAX_REFINEMENTS`, `ADAPTIVE_MAX_LATENCY_MS`, and initial/escalated
`k`. It deduplicates stable `source#chunk_index` identities and stops on
repeated evidence, empty corpus, metadata-filtered no-results, invalid
structured refinement, provider failure, timeout, or budget exhaustion.
Refinement is retrieval-only. The original query remains in the answer prompt
and trace. Existing rewrite, expansion, reranking, filters, freshness, parent
expansion, and compression are composed within each normal retrieval call;
HyDE is intentionally unsupported until an explicit policy is added.

Evidence acceptance combines non-empty evidence, top score, score margin,
source diversity, and lexical query coverage. These are corpus-tuned heuristic
signals, not calibrated raw LLM self-confidence. The additive `trace` response
field contains only typed observable transitions, hit IDs, signals, decisions,
provider/model accounting, and stop reasons; it never contains hidden
chain-of-thought. JSON and SSE use the same controller, while disabled mode
retains the existing single-shot behavior. Issue #72's unbounded iterative
proposal is consolidated into this bounded policy. Evaluation remains the
existing RAGAS/manual-only workflow.

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

## Extractive context compression

Compression is disabled by default (`CONTEXT_COMPRESSION_ENABLED=false`). When
enabled, the exact order is **retrieve -> optional cross-encoder rerank ->
freshness/ranking -> parent-section expansion -> compression -> prompt ->
generation**. It receives the final text, including `expanded_text`, but never
changes retrieval scores or source identity. JSON and SSE share the same engine
prompt path, so they receive identical compressed context behavior.

`localrag/rag/compressor.py` uses deterministic query-term scoring and a Unicode-
aware whitespace token count (`len(re.findall(r"\\S+", text))`). It preserves
ranked context order, source/chunk/parent identity, and records selected character
spans, token/character counts, version, and status under `compression`. Complete
fenced code blocks and contiguous Markdown tables are indivisible; an oversized
unit is omitted instead of being malformed or exceeding a budget. Duplicate
chunks retain distinct chunk identities, while overlapping units remain in source
order.

`CONTEXT_COMPRESSION_PER_CONTEXT_TOKENS` and
`CONTEXT_COMPRESSION_TOTAL_TOKENS` are hard limits, as are the corresponding
character settings. The total token budget must be no larger than
`LLM_CONTEXT_WINDOW_TOKENS` minus `CONTEXT_COMPRESSION_RESERVED_PROMPT_TOKENS`
and `CONTEXT_COMPRESSION_RESERVED_ANSWER_TOKENS`. Empty input, no fitting spans,
and scorer/tokenizer/timeout failures return explicit `no_context` or bounded
`fallback` results; they never silently exceed the budget. The first slice is
extractive only and does not add an LLM summarizer or fabricated citations. See
[ADR 022](adr/022-context-compression-contract.md).
