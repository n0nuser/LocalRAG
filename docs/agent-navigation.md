# Navigation for coding agents

This document explains how to load **useful context quickly** when changing LocalRAG: where code lives, what to read first, and which project rules apply.

## Why this matters

Agents (and humans) move faster when they:

1. **Start from stable anchors** — README, `pyproject.toml`, `.env.example`, and [`architecture.md`](architecture.md) before opening random modules.
2. **Route by symptom** — ingest bugs → `localrag/ingestion/`; HTTP contract → `localrag/api/routers/`; RAG quality → `localrag/rag/` and chunk/embed settings.
3. **Respect the toolchain** — Python **3.13+**; dependencies and commands go through **uv** (`uv sync`, `uv run …`). See [README](../README.md) and [`.cursor/rules/project-setup.mdc`](../.cursor/rules/project-setup.mdc).
4. **Avoid duplicating rules** — Non-obvious coding constraints live in **`.cursor/rules/`** (critical rules, Python style, testing, Grug-style preferences). Read those when editing Python, not a second copy here.

## Read order (minimal)

1. [README](../README.md) — what LocalRAG does, quick start, API entry command.
2. [`pyproject.toml`](../pyproject.toml) — dependencies, script entry `localrag = localrag.cli.app:app`, Ruff/pytest config.
3. [`.env.example`](../.env.example) — canonical env var names and defaults (mirrors `Settings` in `localrag/settings.py`).
4. [architecture.md](architecture.md) — layers, data flow, extension points.
5. The specific file(s) for your task (see table below).

## “I’m changing X — open Y”

| Task | Primary locations |
| --- | --- |
| Environment / defaults | `localrag/settings.py`, `.env.example` |
| FastAPI routes (HTTP only) | `localrag/api/routers/*.py` |
| API request/response OpenAPI models | `localrag/api/schemas.py` |
| API use cases (health, ingest rules, query contexts + SSE, collections including rebuild) | `localrag/api/service.py` |
| API persistence boundary (Chroma collections) | `localrag/api/repository.py` |
| API app factory (lifespan, middleware, error handlers) | `localrag/api/main.py` |
| HTTP ingest path validation (`INGEST_ROOTS`, URL decode) | `localrag/api/service.py`, `localrag/settings.py` (`is_path_allowed`), `localrag/api/exceptions.py` + `main.py` handler |
| DI / shared service instances | `localrag/api/dependencies.py` |
| Log format, levels, request ID | `localrag/logging_config.py`, `localrag/api/middleware.py`, `LOG_LEVEL` in `localrag/settings.py` |
| CLI commands | `localrag/cli/app.py`, `localrag/cli/commands/*.py` |
| Parsing a file type | `localrag/ingestion/parsers/`, `localrag/ingestion/loader.py` |
| Chunk size / overlap | `localrag/ingestion/chunker.py`, `localrag/settings.py` |
| Embeddings / Ollama HTTP for embed | `localrag/ingestion/embedder.py` |
| Ingest orchestration | `localrag/ingestion/service.py` |
| Chroma collection / persist path | `localrag/storage/vector_store.py`, settings |
| Retrieval top-k / query embedding | `localrag/rag/retriever.py`, settings |
| Ollama HTTP request/response shapes | `localrag/ollama/schemas.py` (used by embedder, RAG engine, health, setup) |
| Prompt / answer streaming | `localrag/rag/prompt.py`, `localrag/rag/engine.py` |
| Human Ollama install (not Python) | [ollama.md](ollama.md) |

## Commands (uv)

```bash
uv sync
uv run localrag --help
uv run pytest
uv run ruff format .
uv run ruff check .
```

Pre-commit and contribution workflow: [`.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md).

## External dependencies

- **Ollama** runs outside the repo (CLI or Docker). LocalRAG talks over HTTP using `OLLAMA_BASE_URL` and model env vars.
- **Chroma** data is local filesystem under `CHROMA_PERSIST_PATH` (see `.env.example`).

## When to update this doc

Update [agent-navigation.md](agent-navigation.md) if you add major entry points, move packages, or change the “read order” anchors. Update [architecture.md](architecture.md) if layers, routers, schemas/services/repositories, or ingest/RAG flow change materially. See [AGENTS.md](../AGENTS.md) for the full “documentation maintenance for agents” rule.
