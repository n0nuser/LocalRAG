# Architecture

LocalRAG is a small, layered Python package. Most features touch one layer; cross-cutting behavior lives in `localrag/settings.py`, `localrag/logging_config.py`, and `localrag/api/dependencies.py`, with HTTP lifecycle and middleware in `localrag/api/main.py`.

Configuration is resolved once per execution context by `localrag.settings`: built-in defaults < YAML < `.env` < process environment < explicit CLI `--set` overrides. The API loads `LOCALRAG_CONFIG` during lifespan before cached services are constructed. CLI users pass `--config PATH` before a command. YAML sections (`embedding`, `retrieval`, `generation`, `dataset`, and `evaluation`) map to the single flat `Settings` model; unknown YAML keys and CLI fields fail fast. YAML-relative paths resolve against the configuration file directory, while environment-only settings retain current-working-directory behavior. Environment interpolation uses `${NAME}`. Secrets are accepted from environment sources and redacted from `config-show` snapshots. See [ADR 020](adr/020-structured-configuration.md).

The **HTTP API** uses a basic DDD-style split: **schemas** (`localrag/api/schemas.py`) hold request/response OpenAPI models only; **application services** (`localrag/api/service.py`) implement use cases (health, ingest HTTP rules, query SSE mapping, collection operations); **repositories** (`localrag/api/repository.py`) isolate persistence used by those services (Chroma collections wrapping `VectorStore`); **routers** (`localrag/api/routers/*.py`) stay thin adapters. `HttpMappedError` subclasses (`IngestApiError`, `RagApiError`) and the handler in `main.py` translate validation and RAG failures to HTTP without putting that logic in routers.

## Data flow

```mermaid
flowchart LR
  subgraph inputs
    CLI[CLI Typer]
    API[FastAPI]
  end
  subgraph ingestion
    L[loader + parsers]
    C[chunking contract: fixed / structural / recursive]
    E[EmbeddingProvider]
    VS[(Chroma VectorStore)]
    B[BM25Index]
  end
  subgraph rag
    R[Retriever]
    V[vector search]
    X[BM25 search]
    F[freshness decay]
    P[prompt]
    LLM[Ollama chat HTTP]
  end
  CLI --> L
  API --> L
  L --> C --> E --> VS
  VS --> B
  API --> R
  CLI --> R
  R --> V
  R --> X
  V --> VS
  X --> B
  R --> F
  R --> E
  F --> PARENT[optional parent expansion]
  PARENT --> COMP[optional extractive compression]
  COMP --> P --> LLM
```

  - **Ingest:** files → `loader` / `ingestion/parsers/*` → text → the shared `Chunk` contract (`localrag/ingestion/contract.py`) implemented by fixed, structural, or recursive strategies → the factory-created `EmbeddingProvider` → `VectorStore` (Chroma, persistent path from settings). Contract IDs are deterministic from source, strategy, index, and text; offsets are explicitly absent because current strategies normalize or repack text. Empty input emits no chunks and oversized atomic input is retained with an `oversized` marker. The same provider instance embeds retrieval queries. Collection metadata records provider/model/dimension and rejects incompatible operations; changing the embedding space requires an explicit rebuild. The **HTTP** ingest flow runs path decode, existence checks, and `INGEST_ROOTS` in `localrag/api/service.py` (`ingest_file` / `ingest_directory`), then calls `IngestionService`; optional per-request `embed_model` overrides the configured model and is compatibility-checked. Failures raise `IngestApiError` → JSON in `main.py`. CLI ingests call `IngestionService` directly. `POST /ingest/upload` (`ingest_upload` in `service.py`) takes a multipart file instead of a server path: it validates the extension against `loader.SUPPORTED_EXTENSIONS`, streams it to disk under `UPLOAD_DIR` in 1 MiB chunks while enforcing `UPLOAD_MAX_BYTES` (bypassing `INGEST_ROOTS`, since the server picks the destination), then calls `IngestionService.ingest_file` the same way. See the endpoint's OpenAPI description for upload limitations (no AV scan, extension-only validation, single file per request). For long-running directory ingests, `POST /ingest/directory/async` (`ingest_directory_async` in `service.py`) runs the same path validation synchronously, then submits the actual `IngestionService.ingest_directory` call to the in-process `JobRegistry` (`localrag/api/jobs.py`) and returns `202 {job_id, status: "pending"}` immediately; poll `GET /ingest/jobs/{job_id}` (`get_ingest_job`) for `running` / `done` (with `result`) / `failed` (with `error`). See [ADR 021](adr/021-chunking-strategy-contract.md).
- **Rebuild:** `POST /collections/rebuild` and `localrag collections rebuild` list distinct `source` values in the active collection, drop vectors for missing files, and re-chunk/re-embed remaining paths (optional `embed_model` override). Implemented in `IngestionService.rebuild_collection`.
  - **Query (JSON):** `POST /query` returns a complete `QueryResponse` (answer, sources, latency_ms, model) from `query_json` in `localrag/api/service.py`. Retrieval supports vector-only and hybrid (vector + BM25 with reciprocal-rank fusion), optional bounded query expansion, then applies optional freshness decay based on chunk `ingested_at`. When `ADAPTIVE_ENABLED=true`, `AdaptiveRetrievalPolicy` performs bounded evidence evaluation/escalation/refinement and adds an observable trace; thresholds are corpus-tuned heuristics, not calibrated confidence. JSON and SSE use the same engine policy path. `RAGEngine` generates the answer via its injected `provider` (a `BaseLLMProvider` built by `llm/factory.py::build_provider`, resilience-wrapped), so `LLM_BACKEND` genuinely governs which backend answers `/query` — it is no longer hard-wired to Ollama. Requires `X-API-Key` when `API_KEY` is set.
- **Query (SSE stream):** `POST /query/stream` streams tokens as Server-Sent Events. Retrieval runs synchronously first (`get_query_contexts`) so errors map to HTTP before SSE starts, then tokens are mapped via `iter_query_sse_events`. Token streaming likewise goes through `RAGEngine.provider.stream_from_prompt(...)`, so `LLM_BACKEND` governs the streaming path too.
- **Metrics:** `GET /metrics` exposes Prometheus metrics via `prometheus_client` (router at `localrag/api/routers/metrics.py`). No auth required.

## Package map

| Area | Path | Role |
| --- | --- | --- |
| Settings | `localrag/settings.py` | Single resolved `Settings` model; YAML, `.env`, process env, and CLI override sources, plus redacted snapshots |
| Logging | `localrag/logging_config.py`, `localrag/api/middleware.py` | `configure_logging()`, stderr handler on `localrag.*`, `X-Request-ID` on HTTP requests |
| API wiring | `localrag/api/dependencies.py` | Cached factories: vector store, shared embedding provider, BM25 index, retriever, RAG engine, ingestion service, `ChromaCollectionRepository` |
| HTTP API (transport) | `localrag/api/main.py`, `localrag/api/routers/*` | Lifespan (`configure_logging`), `RequestContextMiddleware` (`X-Request-ID`), global exception + validation handlers + `HttpMappedError`; thin route handlers |
| HTTP API (contracts) | `localrag/api/schemas.py` | Pydantic request/response models and path aliases (OpenAPI) |
| HTTP API (use cases) | `localrag/api/service.py` | Health check, ingest HTTP rules, query JSON (`query_json`) + SSE events, collection list/delete/rebuild orchestration |
| API key auth | `localrag/api/dependencies.py` | `require_api_key` dependency — enforces `X-API-Key` when `API_KEY` env var is set |
| Prometheus metrics | `localrag/api/routers/metrics.py` | `GET /metrics` via `prometheus_client.generate_latest()` |
| HTTP API (persistence) | `localrag/api/repository.py` | `ChromaCollectionRepository` → `VectorStore` for collection list/delete and health’s collection list |
| Background ingest jobs | `localrag/api/jobs.py` | `JobRegistry` — in-memory `ThreadPoolExecutor`-backed job store for `POST /ingest/directory/async`; no persistence across process restarts |
| CLI | `localrag/cli/app.py`, `localrag/cli/commands/*`, `docs/cli.md` | `localrag` Typer entry (`pyproject` `[project.scripts]`); `inspect` is read-only local diagnostics and `benchmark` delegates to `evals.matrix.run_matrix` |
| Ingestion orchestration | `localrag/ingestion/service.py` | `IngestionService`: paths → parse → chunk → embed → upsert |
| File formats | `localrag/ingestion/parsers/*` | pdf (Markdown extraction via pdf-inspector, with OCR fallback via pypdfium2 + pytesseract), docx, markdown, text, code |
| Embedding | `localrag/embedding/`, `localrag/ingestion/embedder.py` | Provider protocol/factory, Ollama **`POST /api/embed`**, optional sentence-transformers backend, and collection identity checks |
| Storage | `localrag/storage/vector_store.py` | Chroma client wrapper |
| RAG | `localrag/rag/retriever.py`, `bm25_index.py`, `engine.py`, `prompt.py` | Hybrid retrieval (vector + BM25), freshness decay reranking, prompt build, LLM call |
| Context compression | `localrag/rag/compressor.py` | Disabled-by-default deterministic extractive compression after parent expansion; preserves retrieval provenance and hard token/character budgets |
| Ollama API models | `localrag/ollama/schemas.py` | Pydantic types + `parse_ollama_json` / `parse_ollama_json_line` for outbound requests and responses |
| LLM abstraction | `localrag/llm/` | `BaseLLMProvider`, Ollama/OpenAI/Anthropic providers, factory, cost estimator |
| Agent | `localrag/agent/service.py`, `localrag/api/routers/agent.py` | Anthropic tool-use agent; `POST /agent/query` |
| Eval | `evals/dataset/`, `evals/metrics.py`, `evals/concurrency.py`, `evals/run_evals.py`, `evals/matrix.py`, `evals/results/`, `evals/compare.py`, `evals/report.py`, `evals/leaderboard.py` | Dataset registry + deterministic/RAGAS metrics, bounded evaluation orchestration, canonical matrix runner, versioned result contract/comparison, offline HTML rendering, and strict leaderboard publication; `localrag eval`, `localrag benchmark`, `localrag eval-compare`, `localrag report`, and `localrag leaderboard` CLI commands |
| Audit log | `localrag/audit.py` | `write_audit_record` — durable local JSONL trail (question, sources, answer, model, latency); disabled by default via `AUDIT_LOG_PATH` |

## LLM abstraction
`localrag/llm/` decouples the RAG engine from a specific model API:

| Path | Role |
| --- | --- |
| `localrag/llm/providers/base.py` | `BaseLLMProvider` ABC with `generate(prompt, context)` / `stream(...)` (context-list contract for direct scripting use) and `generate_from_prompt(prompt)` / `stream_from_prompt(prompt)` (already-built-prompt contract used by `RAGEngine`) |
| `localrag/llm/types.py` | `LLMResponse` dataclass (answer, model, tokens_used, latency_ms, estimated_cost_usd) |
| `localrag/llm/providers/ollama.py` | Ollama HTTP provider (default, local) |
| `localrag/llm/providers/openai_provider.py` | OpenAI chat completions |
| `localrag/llm/providers/anthropic_provider.py` | Anthropic messages API |
| `localrag/llm/resilience.py` | `ResilientProvider` — retry-with-backoff (tenacity) + circuit breaker (pybreaker) wrapping any `BaseLLMProvider` (including the `*_from_prompt` methods); optional fallback provider on sustained failure |
| `localrag/llm/factory.py` | `build_provider(settings)` — selects provider by `LLM_BACKEND` env var; always returns a `ResilientProvider`-wrapped instance |
| `localrag/llm/costs.py` | `estimate_cost_usd(model, tokens)` with prefix-match price table |

`RAGEngine` (`localrag/rag/engine.py`) holds a `provider: BaseLLMProvider` field (injected in `localrag/api/dependencies.py::get_engine` via `build_provider(settings)`) and builds its own citation-rich prompt with `localrag.rag.prompt.build_prompt`, then calls `self.provider.stream_from_prompt(prompt, model=model)` — it no longer talks to Ollama's `/api/chat` directly, so switching `LLM_BACKEND` to `openai` or `anthropic` actually changes what answers `/query` and `/query/stream`.

## Embedding abstraction

`localrag/embedding/` defines the provider-neutral single and batch contract.
`build_embedding_provider` selects Ollama by default or the optional
sentence-transformers backend. Ingestion and `Retriever` receive the same
cached instance. Chroma metadata records provider, model, and vector dimension;
an incompatible runtime fails before vector operations. See [ADR 019](adr/019-embedding-provider-contract.md).

## Agent layer

`localrag/agent/service.py` exposes `run_agent(question, engine, api_key, model)`:

1. Calls `anthropic.messages.create(tools=[search_documents, answer_directly])`.
2. Inspects the `ToolUseBlock` in the response content.
3. For `search_documents`: calls `engine.answer()` and packages sources.
4. For `answer_directly`: returns the agent's answer verbatim.

The `reasoning` field records which path was taken. The router in `localrag/api/routers/agent.py` returns HTTP 503 when `ANTHROPIC_API_KEY` is absent.

## Eval system

`evals/run_evals.py` runs deterministic EM/F1, annotation-backed citation accuracy, hallucination rate, and the existing RAGAS metrics against a registered dataset (`--dataset`/`--version`/`--split`, default `localrag-core` `default`). Results write to `evals/results/` with dataset identity, a content checksum, selected record IDs, per-case status/counts, and environment provenance embedded. The CLI command `uv run localrag eval --offline` delegates to this script. See [docs/evaluation-metrics.md](evaluation-metrics.md), [docs/eval-datasets.md](eval-datasets.md), [docs/reproducibility.md](reproducibility.md), and [evaluation ADRs 011-017](adr/011-evaluation-dataset-contract.md).

`evals/matrix.py` is the canonical cross-configuration benchmark contract. `uv run localrag benchmark` validates and expands versioned dimensions into stable case IDs, runs cases in isolated artifact directories, records structured failures, and continues independent cases. The existing single-run evaluator remains the adapter seam; reports must consume the matrix JSON rather than define another schema. `evals/tracking.py` is an optional best-effort MLflow mirror with a provider-neutral lifecycle boundary; local JSON remains authoritative and tracking is disabled by default. See [ADR 018](adr/018-optional-mlflow-experiment-tracking.md).

`evals/report.py` consumes `MatrixManifest` and `ResultFile` directly and writes one deterministic, self-contained `report.html`. It deliberately omits raw document content and source paths, surfaces malformed inputs and missing values, and marks incompatible dataset identities instead of comparing them. The report is a local inspection view, not experiment tracking or leaderboard publication.

`evals/leaderboard.py` is the strict publication adapter. It validates the
provenance-rich leaderboard artifact contract, rejects incompatible or missing
rows, orders exact model identities deterministically, and writes Markdown plus
optional JSON without running benchmark work. See [benchmark-leaderboard.md](benchmark-leaderboard.md) and [ADR 017](adr/017-strict-leaderboard-publication.md).

## Extension points

- **New file type:** add a parser under `localrag/ingestion/parsers/`, register it via `loader` / parser dispatch (see `localrag/ingestion/loader.py`).
- **PDF OCR behavior:** edit `localrag/ingestion/parsers/pdf.py` and the `OCR_*` settings in `localrag/settings.py`; see [ocr.md](ocr.md) for the Tesseract install requirement.
- **Chunking behavior:** edit `localrag/ingestion/contract.py`, the selected strategy (`chunker.py`, `structural_chunker.py`, or `recursive_chunker.py`), and related knobs in `localrag/settings.py`; see [ADR 021](adr/021-chunking-strategy-contract.md).
- **New HTTP surface:** add schemas in `localrag/api/schemas.py`, application logic in `localrag/api/service.py`, persistence in `localrag/api/repository.py` (if new storage access), thin router in `localrag/api/routers/`, wire DI in `localrag/api/dependencies.py`, include the router in `localrag/api/main.py`.
- **New CLI command:** new module under `localrag/cli/commands/`, register in `localrag/cli/app.py`.
- **New config:** field on `Settings` in `localrag/settings.py`, document in `.env.example`, use via `get_settings()`.
- **Retrieval ranking:** modify `localrag/rag/retriever.py` and `localrag/rag/bm25_index.py` for fusion/decay behavior.
- **Stricter HTTP ingest policy:** adjust checks in `localrag/api/service.py` (`ingest_file` / `ingest_directory`) and/or `is_path_allowed` in `localrag/settings.py`.

## Tests

Tests live under `tests/` (`conftest.py` sets a quiet default `LOG_LEVEL` for pytest). Many cases use stubs or HTTP mocks (e.g. Ollama via respx); run with `uv run pytest`.
