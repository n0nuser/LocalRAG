# Pipeline performance and quality research

> **Dated research snapshot (2026-08-03).** This is a point-in-time investigation, not a maintained map of current behavior. Code, defaults, and measurements have moved on since it was written — verify against the current source and the [docs index](../README.md#documentation) before acting on it.

Stage-by-stage audit of LocalRAG's ingestion and RAG pipeline (parse → chunk →
embed → upsert → retrieve → rerank → generate), grounded in primary sources
(official docs, upstream source, specs) and in measurements taken against this
repo.

**Method.** Every upstream claim carries a URL. Claims I could not tie to a
primary source are marked **Unverified**. Measurements labelled *Measured here*
were taken on this machine against the running `localrag-ollama-1` container
(`nomic-embed-text` 137M F16, `chromadb` 1.5.9, `rank-bm25` 0.2.2) — they are
local observations, not vendor benchmarks.

**Scope note.** PDF extraction (`parse_pdf`) and the structural chunker were
being edited concurrently with this research, so findings that belong to those
two files are stated but not re-measured. Chunking CPU cost is explicitly out of
scope: it measures ~0 ms and is not a bottleneck.

## Ranked findings (impact / effort)

| # | Finding | Type | Impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | Freshness decay overwhelms hybrid RRF ranking | Retrieval quality | High | Low | Verified |
| 2 | `RAG_MIN_CONTEXT_SCORE` is meaningless in hybrid mode | Retrieval quality | High | Low | Verified |
| 3 | Embeddings silently truncated (`truncate` defaults true) | Retrieval quality | Med-High | Low | Verified |
| 4 | BM25 substring boost rescans whole corpus per query | Latency | Medium | Low | Measured here |
| 5 | BM25 full rebuild after every ingest batch | Throughput | Medium | Medium | Measured here |
| 6 | `nomic-embed-text` task prefixes not applied | Retrieval quality | Low-Med | Low | Verified spec / no measured gain |
| 7 | New `httpx.Client` per embed request | Latency | Low-Med | Low | Measured here |
| 8 | Chroma HNSW left at defaults, `ef_search` untuned | Retrieval quality | Low-Med | Low | Verified |
| 9 | Single unbatched `upsert` can exceed Chroma's max batch | Throughput | Low | Low | Verified |
| 10 | Whole-file `read_bytes()` for content hash | Throughput | Low | Low | Measured here |

---

## Stage: parse

### 1) PDF headings from font-size heuristics promote page furniture

**Type:** retrieval quality.

**Generic issue.** Markdown extractors that infer heading levels from font-size
ratios cannot distinguish a real section heading from repeated page furniture
(footers, running heads, page numbers), because the discriminating signal is
position and repetition, not size.

**LocalRAG-specific issue.** ADR 010 already records this as a known limitation:
"Heading detection is heuristic (font-size ratios), so unusual typography can
produce spurious heading levels." The ADR also records that `.pdf` is absent from
`MARKDOWN_EXTENSIONS` (`localrag/ingestion/loader.py:12`), so
`chunk_document()` (`localrag/ingestion/structural_chunker.py:25`) routes PDFs
down `_chunk_non_markdown` and every PDF chunk carries `heading_path=''`. That in
turn disables parent-section expansion for PDFs entirely, since
`_expand_to_parent_section` skips hits with an empty `heading_path`
(`localrag/rag/retriever.py:211-213`).

**What the primary source says.** `pdf-inspector`'s installed API surface offers
no heading-suppression knob — `extract_pages_markdown(path, pages=None)` takes
only a page list, and `PageMarkdown` exposes just `markdown`, `needs_ocr`,
`ocr_reason`, `page` (verified by introspecting the installed package; the
upstream README documents heading detection as "H1-H4 via font size ratios" but
documents no configuration for it —
<https://github.com/firecrawl/pdf-inspector>). The docstring does note that
"Font statistics are computed from the full document so header detection is
consistent across pages."

Suppression therefore has to happen downstream. The library does expose the raw
material for it: `extract_text_with_positions()` returns `TextItem` objects with
`font_size`, `x`, `y`, `height`, `page`, and `is_bold` (verified by
introspection).

**Suggested approach (Unverified — no primary source prescribes this).** The
standard signal for page furniture is *repetition at a stable y-position across
pages*: a heading candidate whose text (or its digit-normalized form) recurs on
most pages at approximately the same vertical offset is furniture, not a
heading. I found no primary specification endorsing a specific threshold, so
treat any concrete cutoff as a tuning parameter to validate on real documents.

**Impact / effort.** Medium impact (unlocks correct `heading_path` for PDFs, and
therefore parent expansion), medium effort. **Risk:** over-aggressive
suppression deletes genuine headings that legitimately repeat (e.g. "Article 1"
in legal texts). This finding *extends* ADR 010's stated follow-up rather than
revising it.

---

## Stage: chunk

**Checked, no issue found.** Chunking CPU cost is ~0 ms and was excluded from
scope by the task. The overlap policy is deliberate and documented: overlap
applies only when `_split_long_paragraph` hard-splits an oversized paragraph,
while packed structural chunks are intentionally disjoint
(`docs/rag-retrieval.md`, "Chunk overlap"). ADR 004 records the structural-vs-
fixed decision. I found no primary source prescribing a universal chunk
size/overlap, and I am not going to invent one — the existing rationale
(boundary-awareness substitutes for overlap) is coherent.

---

## Stage: embed

### 3) Oversized inputs are silently truncated, not rejected

**Type:** retrieval quality (silent data loss).

**Generic issue.** An embedding API that truncates by default will happily return
HTTP 200 and a well-formed vector for input it only partially read. Nothing
downstream can detect that the vector represents a prefix rather than the whole
chunk.

**LocalRAG-specific issue.** `OllamaEmbedRequest`
(`localrag/ollama/schemas.py`) declares only `model` and `input`, and
`_embed_inputs` (`localrag/ingestion/embedder.py:46`) sends exactly those two
fields. `truncate` is never set, so Ollama's default applies.

**What the primary source says.** Ollama's API reference for `POST /api/embed`
documents `truncate` as: "truncates the end of each input to fit within context
length. Returns error if `false` and context length is exceeded. Defaults to
`true`" — <https://docs.ollama.com/api> (also
<https://github.com/ollama/ollama/blob/main/docs/api.md>).

**Measured here.** Against the running server with a ~40 000-character input:

| `truncate` | Result |
|---|---|
| `true` (default — what LocalRAG sends) | HTTP 200, full-length embedding returned |
| `false` | HTTP 400 `{"error":"the input length exceeds the context length"}` |

So the failure is genuinely silent today.

**How exposed is this in practice?** Modest, but non-zero. `/api/show` reports
`nomic-bert.context_length: 2048` with a Modelfile `PARAMETER num_ctx 8192`, and
the Ollama library page lists a "2K context window"
(<https://ollama.com/library/nomic-embed-text>). A default chunk at
`chunk_max_chars=1200` is roughly 300 tokens, comfortably inside that. The
exposure is at the edges: `CHUNK_MAX_CHARS` raised well above the default, or a
`chunking_mode=fixed` misconfiguration.

**Fix.** Send `truncate: false` and let ingestion fail loudly, or keep truncation
but log when input length approaches the limit. **Impact:** medium-high (turns
silent corruption into a visible error). **Effort:** low — one field on
`OllamaEmbedRequest` plus error handling. **Risk:** ingestion starts failing on
documents that previously "succeeded" — which is the point, but it is a
behavioral change worth a release note.

### 6) `nomic-embed-text` task prefixes are not applied

**Type:** retrieval quality.

**Generic issue.** Asymmetric embedding models trained with task instruction
prefixes expect queries and documents to be marked differently. Omitting the
prefixes embeds both into the same undifferentiated space.

**LocalRAG-specific issue.** No prefix is applied anywhere. `embed_texts` sends
chunk text verbatim (`localrag/ingestion/embedder.py:27-42`) and
`Retriever.retrieve` embeds the raw question via `embed_text`
(`localrag/rag/retriever.py:60`). A repo-wide grep for `search_document` /
`search_query` returns only unrelated agent tool names.

**What the primary source says.** The model card for `nomic-embed-text-v1.5`
states that "the text prompt _must_ include a _task instruction prefix_,
instructing the model which task is being performed", listing
`search_document`, `search_query`, `clustering`, `classification` —
<https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>. Ollama serves v1.5 by
default (<https://ollama.com/library/nomic-embed-text>).

**Does Ollama add them for us? No.** `POST /api/show` returns
`template: '{{ .Prompt }}'` with no `system` — a bare passthrough. Whatever the
client sends is what gets embedded. So LocalRAG is, on the letter of the spec,
using the model incorrectly.

**But: measured here, it does not help.** I built a 24-passage documentation-like
corpus with 12 queries each having exactly one gold passage, and compared
prefixed vs unprefixed:

| Variant | recall@1 | MRR | avg top-1 margin |
|---|---|---|---|
| No prefix (current behavior) | 8/12 | 0.794 | 0.1786 |
| With prefixes (per model card) | 8/12 | 0.779 | 0.1584 |

Identical recall@1; MRR and separation margin were marginally *worse* with
prefixes. Two smaller probes agreed. The likely explanation is that adding the
same prefix to every document and every query shifts all vectors similarly and
largely cancels under cosine similarity, and that the effect is real mainly at
larger scale or on harder, more asymmetric corpora than my synthetic one.

**Conclusion — deliberately hedged.** The spec requirement is **verified**; a
retrieval benefit for this corpus is **not observed**. Treat this as a
correctness/compliance item rather than a performance win, and note the migration
cost: prefixes change every vector, so adopting them requires a full
`POST /collections/rebuild`. **Impact:** low-medium. **Effort:** low to
implement, medium to roll out (full re-embed). **Risk:** a rebuild that buys
nothing measurable. Recommend validating on a real corpus before committing.

### 7) A fresh `httpx.Client` is constructed per embed request

**Type:** latency.

**LocalRAG-specific issue.** `_embed_inputs` opens
`with httpx.Client(timeout=self.timeout_seconds) as client:` inside the call
(`localrag/ingestion/embedder.py:55`), so every batch pays connection setup and
teardown. `embed_texts` loops over batches (`:39-41`), so an N-batch file makes N
new clients. The same applies per query at retrieval time via `embed_text`.

**Measured here.** 8 sequential batches of 32 chunks:

| Pattern | Wall time |
|---|---|
| New `Client` per call (current) | 1.976 s |
| Reused `Client` | 1.597 s |

≈ 47 ms/call, ~19% of embed wall time. Modest for ingestion, but it is also on
the per-query path.

**Primary source.** httpx documents that a client enables connection pooling and
reuse across requests, in contrast to top-level per-request APIs —
<https://www.python-httpx.org/advanced/clients/>.

**Impact:** low-medium. **Effort:** low (hold one client on the dataclass).
**Risk:** the embedder becomes stateful and needs deliberate lifecycle/cleanup;
`OllamaEmbedder` is currently an `lru_cache`d singleton
(`localrag/api/dependencies.py`), so a long-lived pooled client fits, but thread
safety across the ingest `ThreadPoolExecutor` (`max_workers=2`,
`localrag/application/jobs.py:48`) should be confirmed.

**Checked, no issue found — embedding batch size.** `embedding_batch_size=32`
(`localrag/settings.py:97`) is already near-optimal. Measured on 128 chunks of
~1200 chars: batch 1 → 23.8 chunks/s, 8 → 98.9, 16 → 134.9, **32 → 151.0**, 64 →
156.3, 128 → 147.0. Raising it to 64 buys ~3%, and the endpoint does accept
arrays ("text or list of text to generate embeddings for",
<https://docs.ollama.com/api>). Not worth changing.

**Checked, no issue found — Ollama concurrency defaults.** `OLLAMA_NUM_PARALLEL`
defaults to 1 and `OLLAMA_MAX_LOADED_MODELS` to "3 * the number of GPUs or 3 for
CPU inference"; `OLLAMA_KEEP_ALIVE` defaults to 5 minutes
(<https://docs.ollama.com/faq>). Because LocalRAG embeds strictly sequentially,
raising `OLLAMA_NUM_PARALLEL` would change nothing until the client itself issues
concurrent requests. Worth noting: the FAQ warns "Parallel request processing for
a given model results in increasing the context size by the number of parallel
requests" — so client-side embed concurrency is the prerequisite, not a server
flag.

---

## Stage: upsert

### 9) A single `upsert` call is not split against Chroma's max batch size

**Type:** throughput / correctness at the tail.

**LocalRAG-specific issue.** `add_chunks` issues exactly one
`self.collection.upsert(...)` for all chunks of a file
(`localrag/storage/vector_store.py:56-61`), with no chunking of the call itself.

**What the primary source says.** Chroma exposes a client-level maximum batch
size; on the installed `chromadb` 1.5.9, `client.get_max_batch_size()` returns
**5461** (verified by calling it). Chroma's public "Add Data" docs do not mention
the limit (<https://docs.trychroma.com/docs/collections/add-data>), so the
installed API is the authority here.

**Reachability.** 5461 chunks at `chunk_max_chars=1200` is ~6.5 MB of extracted
text in a single file. Not typical, but reachable for a large book-length PDF —
and `upload_max_bytes` defaults to 100 MB (`localrag/settings.py:106`), so
nothing upstream forbids it.

**Impact:** low (rare), but the failure mode is a hard error mid-ingest.
**Effort:** low — loop in `add_chunks` using `get_max_batch_size()`. **Risk:**
partial writes if one sub-batch fails; the existing
delete-then-write ordering in `_ingest_one`
(`localrag/ingestion/service.py:220`) already accepts a similar window.

### 10) Content hash reads the whole file into memory

**Type:** throughput / memory.

**LocalRAG-specific issue.** `_file_content_hash` does
`hashlib.sha256(path.read_bytes())` (`localrag/ingestion/service.py:263`),
materializing the entire file. With `upload_max_bytes` at 100 MB, a single
ingest can spike RSS by the full file size.

**Measured here** on a 50 MB file: `read_bytes()` + sha256 = 46 ms with a ~50 MB
peak; streamed 1 MB chunks = 32 ms with a ~1 MB peak.

**Primary source.** `hashlib` objects support incremental `update()`, and Python
documents `file_digest()` as a helper for hashing file objects efficiently —
<https://docs.python.org/3/library/hashlib.html>.

**Impact:** low (faster *and* flat memory). **Effort:** low. **Risk:** none
meaningful — the digest is identical.

**Checked, no issue found — cosine space.** The collection is created with
`hnsw:space: cosine` (`localrag/storage/vector_store.py:26`), matching what
`nomic-embed-text` is trained for. Chroma's default is `l2`
(<https://docs.trychroma.com/docs/collections/configure>), so this was a
deliberate and correct choice, already documented in `docs/rag-retrieval.md`.

---

## Stage: retrieve

### 1) Freshness decay overwhelms RRF ranking in hybrid mode

**Type:** retrieval quality. **This is the highest impact/effort finding.**

**Generic issue.** RRF deliberately discards raw scores and returns values
derived only from rank position. Those values live in a very narrow band. Any
multiplicative post-processing with a wide dynamic range will therefore dominate
the ranking it is supposedly adjusting.

**LocalRAG-specific issue.** `_fuse_results` assigns
`score = weight / (rrf_k + rank)` (`localrag/rag/retriever.py:155,159`), then
`apply_freshness` multiplies that score by
`0.5 ** (age_days / half_life_days)` and re-sorts
(`localrag/rag/retriever.py:189-194`). The two scales are wildly mismatched:

| Quantity | Range |
|---|---|
| RRF score, rank 1 (weight 0.5, `rrf_k=60`) | 0.008197 |
| RRF score, rank 2 | 0.008065 |
| RRF score, rank 20 | 0.006250 |
| Freshness factor, 0 days | 1.0 |
| Freshness factor, 90 days | 0.125 |
| Freshness factor, 365 days | 0.0002 |

Rank 1 and rank 2 differ by **1.6%**. Freshness spans **five thousand-fold** at
defaults (`freshness_half_life_days=30.0`, `localrag/settings.py:119`).

**Computed consequence:** with `rrf_k=60`, a rank-1 document loses to a
perfectly fresh rank-2 document once it is merely **0.70 days old**. A rank-1
document aged 90 days scores 0.001025 — below a fresh *rank-5* document at
0.007692. In hybrid mode, ranking is effectively "sort by ingestion date, break
ties by relevance."

Note this is `ingested_at`, not document authorship date
(`localrag/ingestion/service.py:221,233`) — so a bulk re-ingest resets the
signal for everything, and a freshly re-ingested irrelevant document outranks a
highly relevant one ingested last month.

**What the primary source says.** RRF is defined as
`RRFscore(d) = Σ 1/(k + r(d))` with `k = 60`, introduced by Cormack, Clarke and
Büttcher, SIGIR 2009 (DOI `10.1145/1571941.1572114`, ACM full text paywalled —
<https://dl.acm.org/doi/10.1145/1571941.1572114>). Elasticsearch's first-party
reference reproduces the formula as
`score += 1.0 / (k + rank(result(q), d))` with `rank_constant` defaulting to 60,
and states plainly that "RRF requires no tuning, and the different relevance
indicators do not have to be related to each other" — i.e. the output is
rank-derived and not on a calibrated relevance scale —
<https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion>.

That last property is exactly what makes multiplying RRF output by an unbounded
decay factor unsound: there is no meaningful "50% as relevant" in RRF space.

**Fix options** (Unverified as to which is best — no primary source prescribes
combining recency with RRF):
1. Apply freshness as a **rank-space** adjustment — treat recency as a third
   ranked list fused via RRF — keeping everything on one commensurate scale.
2. **Bound** the decay factor (e.g. floor it at 0.5) so recency can break ties
   but cannot override large relevance gaps.
3. Apply freshness **before** fusion, to each ranker's own scores, so fusion
   still sees ranks.

**Impact:** high. **Effort:** low for option 2, medium for option 1. **Risk:**
changes ranking for every existing hybrid deployment; needs before/after
validation. **This would revise ADR 006**, which specifies multiplying the score
after fusion without discussing the scale interaction with ADR 005's RRF.

### 2) `RAG_MIN_CONTEXT_SCORE` is uncalibratable across retrieval modes

**Type:** retrieval quality (a safety gate that cannot be set correctly).

**LocalRAG-specific issue.** `_is_low_confidence` compares
`contexts[0]["score"]` against `rag_min_context_score`
(`localrag/rag/engine.py:71-78`). But that score means completely different
things per mode:

| Mode | Score formula | Range |
|---|---|---|
| `vector` | `1.0 / (1.0 + distance)` (`retriever.py:128`) | ~0.333 – 1.0 |
| `hybrid` | RRF sum (`retriever.py:155,159`) | ≤ **0.0164** (both lists rank 1) |

A threshold tuned in vector mode (say 0.5) refuses **100% of queries** in hybrid
mode, since the maximum attainable fused score is 0.0164. Freshness decay
(finding 1) then pushes typical hybrid scores lower still.

`docs/rag-retrieval.md` does warn that "there is no universal threshold — tune it
per-corpus", which is honest, but understates the problem: the gate silently
inverts from "off" to "refuse everything" on a `RETRIEVAL_MODE` change alone,
with no error.

**Primary-source grounding.** Same Elasticsearch reference as above: RRF output
is not a calibrated relevance score, so thresholding it directly is not
meaningful.

**Fix.** Either normalize scores to a mode-independent scale before the gate, or
gate on a mode-appropriate signal (e.g. raw top cosine similarity retained
alongside the fused score), or validate the threshold against the active mode at
startup and refuse to boot on an obviously impossible value. **Impact:** high
(silent, total refusal is a severe failure mode). **Effort:** low. **Risk:**
low — the feature is off by default (`rag_min_context_score=0.0`).

### 4) BM25 substring boost rescans and re-lowercases the corpus per query

**Type:** latency.

**LocalRAG-specific issue.** After scoring, `Bm25Index.query` loops over every
document and calls `document.lower()` to test for a raw substring match
(`localrag/rag/bm25_index.py:60-64`). This is O(corpus) per query and repeats
identical `.lower()` work on every call. It is also undocumented in
`docs/rag-retrieval.md` and not covered by any ADR — an unrecorded +1.0 score
bump layered on top of BM25.

**Measured here** at 20 000 chunks:

| Operation | Time |
|---|---|
| `BM25Okapi.get_scores` alone | 26.8 ms |
| `get_scores` + substring boost (current) | 39.6 ms |
| `.lower()` scan alone | 6.8 ms |

The boost is **12.7 ms — 32% of BM25 query time**, and grows linearly with
corpus size.

**Fix.** Precompute lowercased documents once in `refresh()` (trades memory for
time), or drop the boost in favor of proper BM25 scoring. **Impact:** medium.
**Effort:** low. **Risk:** precomputing roughly doubles corpus memory; removing
the boost changes exact-phrase ranking behavior that users may now depend on
(and which motivated ADR 005's exact-token goal).

### 5) BM25 index fully rebuilds after every ingest batch

**Type:** throughput.

**LocalRAG-specific issue.** `ingest_paths` calls `self.bm25_index.refresh()`
once per batch (`localrag/ingestion/service.py:137-138`). The comment there
correctly notes this is deliberate — "Rebuilding the BM25 corpus is O(total
chunks); do it once per batch, not per file" — so the per-file case is already
handled. The residual cost is that `refresh()` re-reads **every chunk in the
store** and re-tokenizes it, even when one small file changed.

**Measured here:**

| Corpus | `get_all_chunks` | `bm25.refresh` | `bm25.query` |
|---|---|---|---|
| 1 000 | 17.3 ms | 57.0 ms | 1.6 ms |
| 5 000 | 89.6 ms | 301.7 ms | 8.6 ms |
| 20 000 | 368.0 ms | 1 330.0 ms | 44.0 ms |

Clean linear scaling. At 100k chunks this extrapolates to ~6.7 s of rebuild after
*any* ingest, plus full-corpus memory materialization.

**Primary source.** `rank_bm25`'s `BM25Okapi` computes corpus statistics in its
constructor and exposes no incremental-add API — verified against the upstream
source, <https://github.com/dorianbrown/rank_bm25>. So incremental update is not
available without either changing library or maintaining the statistics
manually.

**Impact:** medium, and it is the ingestion cost that actually grows with corpus
size (unlike parse/chunk/embed, which are per-file). **Effort:** medium.
**Risk:** hand-maintaining BM25 IDF statistics incrementally is easy to get
subtly wrong; the safer intermediate step is to make the rebuild asynchronous or
debounced rather than incremental.

### 8) Chroma HNSW parameters left at defaults

**Type:** retrieval quality (recall).

**LocalRAG-specific issue.** The collection is created with only
`metadata={"hnsw:space": "cosine"}` (`localrag/storage/vector_store.py:24-27`).
Everything else takes Chroma's defaults.

**What the primary source says.** Chroma documents `space` (default `l2`),
`ef_construction` (100), `ef_search` (100), `max_neighbors` (16), `num_threads`,
`batch_size` (100), `sync_threshold` (1000), `resize_factor` (1.2), and states
that `ef_search`, `num_threads`, `batch_size`, `sync_threshold` and
`resize_factor` "can be modified after creation" —
<https://docs.trychroma.com/docs/collections/configure>.

**Verified here** on `chromadb` 1.5.9. The legacy `metadata={"hnsw:space": ...}`
form still resolves correctly (yielding
`{'space': 'cosine', 'ef_construction': 100, 'ef_search': 100, 'max_neighbors': 16, ...}`)
and emitted no deprecation warning, but the modern `configuration=` form is what
current docs use and is the only way to reach the other knobs:

```python
configuration={"hnsw": {"space": "cosine", "ef_construction": 200,
                        "max_neighbors": 32, "ef_search": 200}}
```

I also confirmed `col.modify(configuration={"hnsw": {"ef_search": 300}})`
succeeds on an existing collection — so **`ef_search` is tunable at runtime with
no re-index**, making it the cheapest available recall lever. `space`,
`ef_construction` and `max_neighbors` are creation-time only, so changing those
requires a rebuild.

**Note.** `ef_search` must be ≥ the number of results requested to avoid
degraded recall; LocalRAG over-fetches up to `rerank_fetch_k=20` or `top_k*2`
(`localrag/rag/retriever.py:43-47`), well under the default 100, so the current
default is not actively harmful — this is headroom, not a bug.

**Impact:** low-medium. **Effort:** low for `ef_search`. **Risk:** higher
`ef_search` trades query latency for recall; `ef_construction`/`M` increases cost
index build time and memory and cannot be changed in place.

**Checked, no issue found — metadata pre-filtering.** The `where=` clause is
passed natively to Chroma so filtering happens before HNSW results are returned
(`localrag/storage/vector_store.py:79`), with a client-side equality check for
the BM25 path (`localrag/rag/retriever.py:22-25,91`). This is the correct split
given `rank_bm25` has no filter concept, and it is already documented.

**Checked, minor — parent-section expansion is N+1.** `_expand_to_parent_section`
issues one `get_chunks_by_heading` per surviving hit
(`localrag/rag/retriever.py:215-217`). Measured at 20 000 chunks: 2.9 ms/call, so
**14.3 ms per query at `top_k=5`** — versus 2.4 ms for the vector query itself.
Real but small next to LLM generation; batching all hits into one `$or` query
would remove it. Low impact, low effort.

---

## Stage: rerank

**Checked, no issue found — batching.** `CrossEncoderReranker.rerank` calls
`predict(pairs)` without `batch_size` (`localrag/rag/reranker.py:38`), so
sentence-transformers' default of **32** applies
(<https://www.sbert.net/docs/package_reference/cross_encoder/model.html>). Since
`rerank_fetch_k` defaults to 20 (`localrag/settings.py:129`), the entire
candidate set fits in a single batch. Explicitly passing `batch_size` would
change nothing unless `rerank_fetch_k` is raised above 32 — worth a comment, not
a fix.

**Checked, no issue found — truncation.** `cross-encoder/ms-marco-MiniLM-L-6-v2`
has `max_position_embeddings: 512` (verified from the model repo's `config.json`,
<https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2>), and the loaded
model reports `max_seq_length = 512`. Measured with the real tokenizer: a query
plus a 1200-char chunk tokenizes to **251 tokens**, and a deliberately
token-dense 1200-char chunk to **371** — both inside the limit. Default
`chunk_max_chars=1200` is safe. This would only become a problem if
`CHUNK_MAX_CHARS` were roughly doubled.

**Correct ordering, confirmed.** Reranking runs on the raw candidate list before
freshness and expansion (`localrag/rag/retriever.py:97-102`), matching ADR 008.
Worth flagging that finding 1 partially undermines this: the reranker carefully
orders candidates, and `apply_freshness` then re-sorts them by a factor with a
much wider range, discarding much of the cross-encoder's work. `rerank_score` is
preserved on each context (`reranker.py:40`) but is not what the final sort uses.

---

## Stage: generate

**Checked, no issue found — prompt construction.** `build_prompt`
(`localrag/rag/prompt.py`) is a straightforward join with per-context source
attribution, preferring `expanded_text` when present. No redundant work.

**Observation — no context-length budgeting (Unverified impact).** `build_prompt`
concatenates every context with no token budget. With `parent_expansion_enabled`
(default true) each of `top_k=5` hits can expand to a whole section, so prompt
size is unbounded in principle. Ollama's documented default context length is
4096 tokens, settable via `OLLAMA_CONTEXT_LENGTH` or `num_ctx`
(<https://docs.ollama.com/faq>), and `OllamaChatRequest`
(`localrag/ollama/schemas.py`) sends no `options`, so `num_ctx` is never set by
LocalRAG. Chat-side overflow behavior is handled server-side and I did not
measure it here, so I am not claiming a specific failure — but a prompt that
exceeds the server's context is a plausible silent-truncation path analogous to
the verified embed one (finding 3). Worth measuring before acting.

---

## Summary: checked, no issue found

Recording these to prevent re-investigation.

- **Embedding batch size (32)** — measured near-optimal; 64 buys ~3%.
- **`/api/embed` array batching** — supported and already used correctly.
- **Cross-encoder batching** — candidate set (20) fits the default batch (32).
- **Cross-encoder truncation** — 1200-char chunks are 251–371 tokens vs a
  512 limit.
- **Cosine space** — correctly set, matching `nomic-embed-text`.
- **Rerank ordering vs ADR 008** — implementation matches the ADR.
- **Metadata pre-filtering** — native `where=` on the vector path is correct.
- **BM25 rebuild once per batch, not per file** — already optimized.
- **Chunking CPU cost** — ~0 ms, not a bottleneck.
- **Chunk overlap policy** — deliberate and documented, not an oversight.
- **Legacy `hnsw:` metadata key** — still functional on chromadb 1.5.9, no
  deprecation warning observed.

## Findings that would revise an existing ADR

- **ADR 006 (freshness decay)** — finding 1. The ADR specifies multiplying the
  post-fusion score by an exponential decay factor without addressing that ADR
  005's RRF output occupies a ~1.6%-wide band. The two decisions are individually
  reasonable and jointly produce recency-dominated ranking.
- **ADR 010 (pdf-inspector)** — finding 2 in the parse stage extends the
  already-recorded follow-up on PDF Markdown chunking with the spurious-heading
  problem and the `TextItem` position data available to address it.

## Explicitly unverified

Listed separately so nothing here is mistaken for a sourced claim.

- The specific footer-suppression heuristic (repetition at stable y-position).
  No primary source prescribes it; thresholds need empirical tuning.
- Which of the three freshness/RRF remedies is best. The scale mismatch is
  arithmetic and verified; the remedy is a design choice.
- Prompt context-length overflow at generation time — plausible by analogy to
  the verified embed truncation, but not measured.
- Whether task prefixes help at larger corpus scale. Measured neutral-to-slightly-
  negative on a 24-passage corpus; the spec requirement is verified, the benefit
  is not.
