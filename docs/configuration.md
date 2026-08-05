# Configuration

Every setting resolves into one immutable `Settings` object (`localrag/settings.py`).
Environment variable names are the uppercased field names, so `rag_top_k` is
`RAG_TOP_K`. [`.env.example`](../.env.example) is the canonical env list and
[`config.example.yaml`](../config.example.yaml) shows the structured form.

## Precedence

Highest wins:

1. Explicit CLI `--set FIELD=VALUE`
2. Process environment
3. `.env`
4. YAML selected with `--config` or `LOCALRAG_CONFIG`
5. Built-in defaults

YAML is a **base layer that the environment overrides**, not an override itself.
A stale `OLLAMA_LLM_MODEL` in `.env` silently wins over the same value in an
explicit `--config` file — check with `config-show` when a YAML value appears to
be ignored.

```bash
cp .env.example .env

uv run localrag --config config.yaml ingest ./documents
uv run localrag --config config.yaml --set rag_top_k=10 config-show
LOCALRAG_CONFIG=./config.yaml uv run uvicorn localrag.api.main:app
```

YAML is strict: it accepts only the `embedding`, `retrieval`, `generation`,
`dataset`, and `evaluation` sections, rejects unknown keys, interpolates
`${ENV_NAME}` from the environment, and resolves relative paths against the YAML
file's own directory. Unknown `--set` fields fail with `ConfigError`. Invalid or
missing selected files fail before any service is constructed.

`config-show` prints a deterministic resolved snapshot with API keys redacted and
paths shown as `<path>`; document contents and the full environment are never
included. Keep credentials in the environment, never in YAML.

## Core

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | _(empty)_ | Require `X-API-Key` header; empty disables auth |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API bind address and port |
| `LOG_LEVEL` | `INFO` | JSON in production, colored in a TTY |
| `TENANT_ID` | _(empty)_ | Optional tenant tag on records |

## Models and providers

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_BACKEND` | `ollama` | `ollama`, `openai`, or `anthropic` |
| `LLM_FALLBACK_BACKEND` | _(empty)_ | Different provider used after the primary circuit opens |
| `LLM_TEMPERATURE` / `LLM_SEED` | _(unset)_ | Unset keeps each model's own defaults; set both for reproducible answers |
| `LLM_TIMEOUT_SECONDS` | `180.0` | Generation request timeout |
| `LLM_RETRY_MAX_ATTEMPTS` | `3` | Retry attempts per provider call |
| `LLM_CIRCUIT_FAIL_MAX` | `5` | Failures before the circuit opens |
| `LLM_CIRCUIT_RESET_TIMEOUT_SECONDS` | `30.0` | Circuit half-open delay |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `gemma3:4b` | Chat model for the Ollama backend |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | _(empty)_ / `gpt-4o-mini` | Required for the `openai` backend |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | _(empty)_ / `claude-haiku-4-5` | Required for the `anthropic` backend |
| `AGENT_MODEL` | `claude-haiku-4-5` | Model backing `POST /agent/query` |

Unsupported provider names, and selecting the same fallback as the primary, fail
during settings validation. Query responses, SSE final events, audit records, and
evaluation snapshots all record the effective provider and model.

## Embeddings

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or optional `sentence-transformers` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_MODEL` | _(empty)_ | Provider model; falls back to `OLLAMA_EMBED_MODEL` |
| `SENTENCE_TRANSFORMERS_MODEL` | `all-MiniLM-L6-v2` | Model for the optional local provider |
| `EMBEDDING_BATCH_SIZE` | `32` | Embeddings per request |
| `EMBEDDING_TIMEOUT_SECONDS` | `120.0` | Embedding request timeout |
| `EMBEDDING_CACHE_ENABLED` | `false` | Reuse embeddings across ingests |
| `EMBEDDING_CACHE_PATH` | `./data/embedding-cache` | Cache location |
| `EMBEDDING_CACHE_MAX_ENTRIES` / `_MAX_BYTES` | `10000` / `1000000000` | Cache bounds |

Changing the embedding model requires rebuilding collections; compatibility is
enforced by the vector store ([ADR 019](adr/019-embedding-provider-contract.md)).

## Storage and ingestion

| Variable | Default | Description |
| --- | --- | --- |
| `CHROMA_PERSIST_PATH` | `./data/chroma` | Vector store directory |
| `CHROMA_COLLECTION_NAME` | `localrag` | Collection name |
| `CHUNKING_MODE` | `structural` | `structural` or `fixed` |
| `CHUNK_MAX_CHARS` / `CHUNK_MIN_CHARS` | `1200` / `200` | Structural chunk budget and merge floor |
| `CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` | `512` / `150` | Fixed-mode window and overlap |
| `INGEST_RECURSIVE` | `true` | Default for directory ingest |
| `INGEST_ROOTS` | _(empty)_ | Allow-list for HTTP ingest paths; empty disables the restriction |
| `MAX_PENDING_INGEST_JOBS` | `10` | Background ingest queue bound |
| `UPLOAD_DIR` | `./data/uploads` | Where `POST /ingest/upload` writes |
| `UPLOAD_MAX_BYTES` | `100000000` | Max accepted upload size |
| `UPLOAD_RETENTION_SECONDS` | `0.0` | `0` treats uploads as temporary artifacts |
| `UPLOAD_QUOTA_BYTES` | `1000000000` | Total upload directory budget |
| `OCR_ENABLED` | `true` | OCR fallback for scanned PDF pages |
| `OCR_LANGUAGE` / `OCR_MIN_CHARS_PER_PAGE` | `eng` / `20` | Tesseract language and trigger threshold |

`POST /ingest/upload` bypasses `INGEST_ROOTS` because the server chooses the
destination. See [data-lifecycle.md](data-lifecycle.md) and [ocr.md](ocr.md).

## Retrieval and generation

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (vector + BM25) or `vector` |
| `RETRIEVER_PLUGIN` | `builtin` | Named third-party retriever |
| `RRF_K` | `60` | Reciprocal rank fusion smoothing constant |
| `FRESHNESS_HALF_LIFE_DAYS` | `30.0` | Recency decay half-life; `0` disables |
| `FRESHNESS_WEIGHT` | `0.15` | Recency share of the relevance budget |
| `PARENT_EXPANSION_ENABLED` | `true` | Expand hits to their full section |
| `RAG_MIN_CONTEXT_SCORE` | `0.0` | Refuse to generate below this score; `0` disables |
| `RAG_SYSTEM_PROMPT` | _(built-in)_ | System message for the answering model |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking (`rerank` extra) |
| `RERANK_MODEL` / `RERANK_FETCH_K` | `ms-marco-MiniLM-L-6-v2` / `20` | Rerank model and candidate depth |
| `QUERY_REWRITE_ENABLED` | `false` | Bounded query rewriting |
| `QUERY_EXPANSION_ENABLED` | `false` | Bounded query expansion |
| `HYDE_ENABLED` | `false` | Hypothetical-document retrieval experiment |
| `CONTEXT_COMPRESSION_ENABLED` | `false` | Deterministic context compression |
| `ADAPTIVE_ENABLED` | `false` | Bounded adaptive retrieval policy |
| `QUERY_CACHE_TTL_SECONDS` | `0.0` | Query cache TTL; `0` disables |
| `LLM_CONTEXT_WINDOW_TOKENS` | `4096` | Budget used by compression planning |

Ranking math and the full option set for each experiment live in
[rag-retrieval.md](rag-retrieval.md); the bounded experiments each have an ADR
(HyDE [025](adr/025-hyde-retrieval-experiment.md), compression
[022](adr/022-context-compression-contract.md), adaptive
[023](adr/023-bounded-adaptive-retrieval.md)).

## Audit and observability

| Variable | Default | Description |
| --- | --- | --- |
| `AUDIT_LOG_PATH` | _(empty)_ | Enables query audit logging when set |
| `AUDIT_LOG_MAX_BYTES` | `10000000` | Rotation threshold |
| `AUDIT_LOG_RETENTION_SECONDS` | `2592000.0` | 30-day default retention |
| `AUDIT_LOG_METADATA_ONLY` / `_REDACT_CONTENT` | `false` | Reduce what is recorded |
| `OTEL_ENABLED` | `false` | OpenTelemetry tracing (`observability` extra) |
| `OTEL_EXPORTER_ENDPOINT` | `http://localhost:4318` | OTLP endpoint |
| `OTEL_SERVICE_NAME` / `OTEL_SAMPLE_RATE` | `localrag` / `1.0` | Service identity and sampling |
| `OTEL_CAPTURE_CONTENT` | `false` | Off by default for privacy |
| `LANGFUSE_ENABLED` | `false` | Separate optional adapter boundary |
| `EVAL_SEED` | `42` | Default evaluation seed |

See [observability.md](observability.md) for OTLP setup and privacy defaults,
and [data-lifecycle.md](data-lifecycle.md) for audit retention.
