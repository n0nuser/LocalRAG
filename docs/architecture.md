# Architecture

Hermit is a small, layered Python package. Most features touch one layer; cross-cutting behavior lives in `hermit/settings.py`, `hermit/logging_config.py`, and `hermit/api/dependencies.py`, with HTTP lifecycle and middleware in `hermit/api/main.py`.

The **HTTP API** uses a basic DDD-style split: **schemas** (`hermit/api/schemas.py`) hold request/response OpenAPI models only; **application services** (`hermit/api/service.py`) implement use cases (health, ingest HTTP rules, query SSE mapping, collection operations); **repositories** (`hermit/api/repository.py`) isolate persistence used by those services (Chroma collections wrapping `VectorStore`); **routers** (`hermit/api/routers/*.py`) stay thin adapters. `IngestApiError` and its handler in `main.py` translate ingest validation failures to HTTP without putting that logic in routers.

## Data flow

```mermaid
flowchart LR
  subgraph inputs
    CLI[CLI Typer]
    API[FastAPI]
  end
  subgraph ingestion
    L[loader + parsers]
    C[chunker]
    E[OllamaEmbedder]
    VS[(Chroma VectorStore)]
  end
  subgraph rag
    R[Retriever]
    P[prompt]
    LLM[Ollama chat HTTP]
  end
  CLI --> L
  API --> L
  L --> C --> E --> VS
  API --> R
  CLI --> R
  R --> VS
  R --> E
  R --> P --> LLM
```

- **Ingest:** files → `loader` / `ingestion/parsers/*` → text → `chunker` → `OllamaEmbedder` → `VectorStore` (Chroma, persistent path from settings). The **HTTP** ingest flow runs path decode, existence checks, and `INGEST_ROOTS` in `hermit/api/service.py` (`ingest_file` / `ingest_directory`), then calls `IngestionService`; failures raise `IngestApiError` → JSON in `main.py`. CLI ingests call `IngestionService` directly.
- **Query:** question → `Retriever` (embed query, query Chroma) → `build_prompt` → Ollama **`POST /api/chat`** (streaming in API/CLI as implemented). The **HTTP** route delegates streaming event shaping to `hermit/api/service.py` (`iter_query_sse_events`).

## Package map

| Area | Path | Role |
| --- | --- | --- |
| Settings | `hermit/settings.py` | `Settings` + `get_settings()`; env vars from `.env` (includes `log_level`) |
| Logging | `hermit/logging_config.py`, `hermit/api/middleware.py` | `configure_logging()`, stderr handler on `hermit.*`, `X-Request-ID` on HTTP requests |
| API wiring | `hermit/api/dependencies.py` | Cached factories: vector store, embedder, retriever, RAG engine, ingestion service, `ChromaCollectionRepository` |
| HTTP API (transport) | `hermit/api/main.py`, `hermit/api/routers/*` | Lifespan (`configure_logging`), `RequestContextMiddleware` (`X-Request-ID`), global exception + validation handlers + `IngestApiError`; thin route handlers |
| HTTP API (contracts) | `hermit/api/schemas.py` | Pydantic request/response models and path aliases (OpenAPI) |
| HTTP API (use cases) | `hermit/api/service.py` | Health check, ingest HTTP rules, query SSE events, collection list/delete orchestration |
| HTTP API (persistence) | `hermit/api/repository.py` | `ChromaCollectionRepository` → `VectorStore` for collection list/delete and health’s collection list |
| CLI | `hermit/cli/app.py`, `hermit/cli/commands/*` | `hermit` Typer entry (`pyproject` `[project.scripts]`) |
| Ingestion orchestration | `hermit/ingestion/service.py` | `IngestionService`: paths → parse → chunk → embed → upsert |
| File formats | `hermit/ingestion/parsers/*` | pdf, docx, markdown, text, code |
| Chunking / embed | `hermit/ingestion/chunker.py`, `hermit/ingestion/embedder.py` | Local text splits; Ollama **`POST /api/embed`** (see `hermit/ollama/schemas.py`) |
| Storage | `hermit/storage/vector_store.py` | Chroma client wrapper |
| RAG | `hermit/rag/retriever.py`, `engine.py`, `prompt.py` | Retrieve top-k, build prompt, call LLM |
| Ollama API models | `hermit/ollama/schemas.py` | Pydantic types + `parse_ollama_json` / `parse_ollama_json_line` for outbound requests and responses |

## Extension points

- **New file type:** add a parser under `hermit/ingestion/parsers/`, register it via `loader` / parser dispatch (see `hermit/ingestion/loader.py`).
- **New HTTP surface:** add schemas in `hermit/api/schemas.py`, application logic in `hermit/api/service.py`, persistence in `hermit/api/repository.py` (if new storage access), thin router in `hermit/api/routers/`, wire DI in `hermit/api/dependencies.py`, include the router in `hermit/api/main.py`.
- **New CLI command:** new module under `hermit/cli/commands/`, register in `hermit/cli/app.py`.
- **New config:** field on `Settings` in `hermit/settings.py`, document in `.env.example`, use via `get_settings()`.
- **Stricter HTTP ingest policy:** adjust checks in `hermit/api/service.py` (`ingest_file` / `ingest_directory`) and/or `is_path_allowed` in `hermit/settings.py`.

## Tests

Tests live under `tests/` (`conftest.py` sets a quiet default `LOG_LEVEL` for pytest). Many cases use stubs or HTTP mocks (e.g. Ollama via respx); run with `uv run pytest`.
