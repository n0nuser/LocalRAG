# Issues and fixes: Reddit retrieval thread vs LocalRAG

## 1) Chunking that breaks structure

### Generic issue

Fixed-size chunk windows often split logical units (table rows, question-answer
pairs, code blocks), so the model receives partial facts and confidently fills
the gaps.

### LocalRAG-specific issue

Before this change:

- `localrag/ingestion/chunker.py` used character windows only.
- `localrag/ingestion/parsers/markdown.py` flattened markdown body into plain
  text before chunking.

### Fix shipped

- Added `localrag/ingestion/structural_chunker.py`.
- Default chunking mode is now structural (`CHUNKING_MODE=structural`), with
  fixed fallback (`CHUNKING_MODE=fixed`).
- Chunk metadata now includes `heading_path` and `chunk_type`.

## 2) Pure vector retrieval misses exact strings

### Generic issue

Vector similarity works for semantic matching but is weak for exact tokens
(error codes, SKUs, versions, identifiers).

### LocalRAG-specific issue

`localrag/rag/retriever.py` only queried Chroma vectors. A query like
`ERR_QUIC_PROTOCOL_ERROR` could lose to semantically similar but wrong chunks.

### Fix shipped

- Added `rank-bm25` and `localrag/rag/bm25_index.py`.
- Added `VectorStore.get_all_chunks()` to build BM25 corpus from current store.
- `Retriever` now supports `RETRIEVAL_MODE=hybrid` with reciprocal rank fusion
  across vector and BM25 candidates.

## 3) Ranking ignored freshness

### Generic issue

Without time decay, stale content can outrank newer policy or release notes
despite being semantically similar.

### LocalRAG-specific issue

`ingested_at` was already written in ingestion metadata, but retrieval scores
did not use it.

### Fix shipped

- Added `FRESHNESS_HALF_LIFE_DAYS` setting (default `30.0`, `0` disables decay).
- Retriever now multiplies base score by an exponential recency factor.
- Returned contexts include `ingested_at` and `freshness_factor` for debugging.

## Deferred items (not part of this implementation)

- Metadata filter-cardinality auditing (no active metadata pre-filter pipeline in
  current LocalRAG retrieval path).
- Tool-state checkpointing after external actions (agent orchestration concern).
- Fully agentic iterative retrieval loops beyond the current tool-use flow.
