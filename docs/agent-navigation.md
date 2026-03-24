# Navigation for coding agents

This document explains how to load **useful context quickly** when changing Hermit: where code lives, what to read first, and which project rules apply.

## Why this matters

Agents (and humans) move faster when they:

1. **Start from stable anchors** — README, `pyproject.toml`, `.env.example`, and [`architecture.md`](architecture.md) before opening random modules.
2. **Route by symptom** — ingest bugs → `hermit/ingestion/`; HTTP contract → `hermit/api/routers/`; RAG quality → `hermit/rag/` and chunk/embed settings.
3. **Respect the toolchain** — Python **3.13+**; dependencies and commands go through **uv** (`uv sync`, `uv run …`). See [README](../README.md) and [`.cursor/rules/project-setup.mdc`](../.cursor/rules/project-setup.mdc).
4. **Avoid duplicating rules** — Non-obvious coding constraints live in **`.cursor/rules/`** (critical rules, Python style, testing, Grug-style preferences). Read those when editing Python, not a second copy here.

## Read order (minimal)

1. [README](../README.md) — what Hermit does, quick start, API entry command.
2. [`pyproject.toml`](../pyproject.toml) — dependencies, script entry `hermit = hermit.cli.app:app`, Ruff/pytest config.
3. [`.env.example`](../.env.example) — canonical env var names and defaults (mirrors `Settings` in `hermit/config.py`).
4. [architecture.md](architecture.md) — layers, data flow, extension points.
5. The specific file(s) for your task (see table below).

## “I’m changing X — open Y”

| Task | Primary locations |
| --- | --- |
| Environment / defaults | `hermit/config.py`, `.env.example` |
| FastAPI routes, request/response shapes | `hermit/api/routers/*.py`, `hermit/api/main.py` |
| DI / shared service instances | `hermit/api/dependencies.py` |
| CLI commands | `hermit/cli/app.py`, `hermit/cli/commands/*.py` |
| Parsing a file type | `hermit/ingestion/parsers/`, `hermit/ingestion/loader.py` |
| Chunk size / overlap | `hermit/ingestion/chunker.py`, `hermit/config.py` |
| Embeddings / Ollama HTTP for embed | `hermit/ingestion/embedder.py` |
| Ingest orchestration | `hermit/ingestion/service.py` |
| Chroma collection / persist path | `hermit/storage/vector_store.py`, settings |
| Retrieval top-k / query embedding | `hermit/rag/retriever.py`, settings |
| Prompt / answer streaming | `hermit/rag/prompt.py`, `hermit/rag/engine.py` |
| Human Ollama install (not Python) | [ollama.md](ollama.md) |

## Commands (uv)

```bash
uv sync
uv run hermit --help
uv run pytest
uv run ruff format .
uv run ruff check .
```

Pre-commit and contribution workflow: [`.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md).

## External dependencies

- **Ollama** runs outside the repo (CLI or Docker). Hermit talks over HTTP using `OLLAMA_BASE_URL` and model env vars.
- **Chroma** data is local filesystem under `CHROMA_PERSIST_PATH` (see `.env.example`).

## When to update this doc

Update [agent-navigation.md](agent-navigation.md) if you add major entry points, move packages, or change the “read order” anchors. Update [architecture.md](architecture.md) if layers, routers, or ingest/RAG flow change materially.
