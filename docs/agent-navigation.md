# Navigation for coding agents

This document explains how to load **useful context quickly** when changing LocalRAG: where code lives, what to read first, and which project rules apply.

## Why this matters

Agents (and humans) move faster when they:

1. **Start from stable anchors** — README, `pyproject.toml`, `.env.example`, and [`architecture.md`](architecture.md) before opening random modules.
2. **Route by symptom** — ingest bugs → `localrag/ingestion/`; HTTP contract → `localrag/api/routers/`; RAG quality → `localrag/rag/` and chunk/embed settings.
3. **Respect the toolchain** — Python **3.13+**; dependencies and commands go through **uv** (`uv sync --locked`, `uv run …`). See [README](../README.md) and [`.cursor/rules/project-setup.mdc`](../.cursor/rules/project-setup.mdc).
4. **Avoid duplicating rules** — Non-obvious coding constraints live in **`.cursor/rules/`** (critical rules, Python style, testing, Grug-style preferences). Read those when editing Python, not a second copy here.

## Read order (minimal)

1. [README](../README.md) — what LocalRAG does, quick start, API entry command.
2. [ROADMAP](../ROADMAP.md) — current milestone map, issue dependencies, and contributor selection guidance.
3. [`pyproject.toml`](../pyproject.toml) — dependencies, script entry `localrag = localrag.cli.app:app`, Ruff/pytest config.
4. [`.env.example`](../.env.example) — canonical env var names and defaults (mirrors `Settings` in `localrag/settings.py`); [`config.example.yaml`](../config.example.yaml) shows structured configuration.
5. [architecture.md](architecture.md) — layers, data flow, extension points.
6. [deployment.md](deployment.md) — Kubernetes persistence, dependency, probe, and security contract.
7. The specific file(s) for your task (see table below).

For contributor workflows, read [`Taskfile.yml`](../Taskfile.yml) and the
Taskfile section in [CONTRIBUTING](../.github/CONTRIBUTING.md). It is the thin
wrapper contract for `uv`, CLI, test, lint, benchmark, report, and Compose
commands; its Docker tasks intentionally do not load the host-specific WSL2
override unless `COMPOSE_OVERRIDE` is supplied.

## “I’m changing X — open Y”

| Task | Primary locations |
| --- | --- |
| Environment / defaults | `localrag/settings.py`, `.env.example`, `config.example.yaml`, `docs/adr/020-structured-configuration.md` |
| FastAPI routes (HTTP only) | `localrag/api/routers/*.py` |
| API request/response OpenAPI models | `localrag/api/schemas.py` |
| API use cases (health, ingest rules, query JSON + SSE, collections including rebuild) | `localrag/api/service.py` |
| API persistence boundary (Chroma collections) | `localrag/api/repository.py` |
| API app factory (lifespan, middleware, error handlers) | `localrag/api/main.py` |
| HTTP ingest path validation (`INGEST_ROOTS`, URL decode) | `localrag/api/service.py`, `localrag/settings.py` (`is_path_allowed`), `localrag/api/exceptions.py` + `main.py` handler |
| HTTP multipart file upload ingest (`POST /ingest/upload`) | `localrag/api/routers/ingest.py` (`ingest_upload`, Swagger limitations in `_UPLOAD_DESCRIPTION`), `localrag/api/service.py` (`ingest_upload`, cleanup), upload lifecycle settings in `localrag/settings.py`, [data-lifecycle.md](data-lifecycle.md) |
| Query audit lifecycle | `localrag/audit.py`, audit lifecycle settings in `localrag/settings.py`, [data-lifecycle.md](data-lifecycle.md) |
| Background ingest jobs | `localrag/api/jobs.py`, `localrag/api/routers/ingest.py` (async routes), `localrag/api/service.py` (`ingest_directory_async`, `get_ingest_job`) |
| DI / shared service instances | `localrag/api/dependencies.py` |
| Log format, levels, request ID | `localrag/logging_config.py`, `localrag/api/middleware.py`, `LOG_LEVEL` in `localrag/settings.py` |
| Optional tracing / observability | `localrag/observability/tracing.py`, `OTEL_*` in `localrag/settings.py`, [observability.md](observability.md), [ADR 030](adr/030-optional-otel-observability-boundary.md) |
| API key auth | `localrag/api/dependencies.py` (`require_api_key`), `API_KEY` in `localrag/settings.py` |
| Prometheus metrics endpoint | `localrag/api/routers/metrics.py` |
| LLM provider abstraction | `localrag/llm/providers/`, `localrag/llm/factory.py` |
| Cost estimation | `localrag/llm/costs.py` |
| Agent tool-use (search_documents / answer_directly) | `localrag/agent/service.py`, `localrag/api/routers/agent.py` |
| Architecture decisions | `docs/adr/` |
| Roadmap status and milestone selection | `ROADMAP.md`, GitHub issues and milestones, `scripts/validate_roadmap.py` |
| CLI commands | `localrag/cli/app.py`, `localrag/cli/commands/*.py`, [cli.md](cli.md) |
| Structured configuration | `localrag/settings.py`, `localrag/cli/app.py`, `localrag/api/main.py`, `config.example.yaml` |
| Parsing a file type | `localrag/ingestion/parsers/`, `localrag/ingestion/loader.py`; optional anydoc formats use `localrag/ingestion/parsers/anydoc.py` and the `anydoc` extra |
| PDF OCR (scanned/image-only pages) | `localrag/ingestion/parsers/pdf.py`, `OCR_*` in `localrag/settings.py`, [ocr.md](ocr.md) |
| Chunking strategy and boundaries | `localrag/ingestion/contract.py`, `localrag/ingestion/structural_chunker.py`, `localrag/ingestion/chunker.py`, `localrag/ingestion/recursive_chunker.py`, `localrag/settings.py`, `docs/adr/021-chunking-strategy-contract.md` |
| Embeddings / provider contract and factory | `localrag/embedding/`, `localrag/ingestion/embedder.py` |
| Ingestion embedding cache | `localrag/embedding/cache.py`, embedding cache settings, [ADR 024](adr/024-embedding-cache-contract.md), `benchmarks/embedding_cache_benchmark.py` |
| Ingest orchestration | `localrag/ingestion/service.py` |
| Chroma collection / persist path | `localrag/storage/vector_store.py`, settings |
| Retrieval mode / hybrid ranking / freshness decay / HyDE experiment | `localrag/rag/retriever.py`, `localrag/rag/hyde.py`, `localrag/rag/bm25_index.py`, `localrag/settings.py`, [ADR 025](adr/025-hyde-retrieval-experiment.md) |
| Retriever plugin contract / discovery | `localrag/plugins/retriever.py`, [plugin-author-guide.md](plugin-author-guide.md), [ADR 032](adr/032-retriever-plugin-contract.md) |
| Bounded adaptive retrieval policy / trace | `localrag/rag/adaptive.py`, `localrag/rag/engine.py`, adaptive settings, [ADR 023](adr/023-bounded-adaptive-retrieval.md) |
| Context compression contract and budgets | `localrag/rag/compressor.py`, `localrag/rag/engine.py`, `localrag/settings.py`, `docs/adr/022-context-compression-contract.md` |
| Ollama HTTP request/response shapes | `localrag/ollama/schemas.py` (used by embedder, RAG engine, health, setup) |
| Embedding collection compatibility | `localrag/storage/vector_store.py`, [ADR 019](adr/019-embedding-provider-contract.md), embedding settings in `localrag/settings.py` |
| Generation sampling (temperature/seed) | `localrag/ollama/schemas.py` (`OllamaChatOptions`), `localrag/llm/providers/ollama.py`, `LLM_TEMPERATURE` / `LLM_SEED` in `localrag/settings.py` |
| Eval suite / metric contracts / bounded RAGAS scoring | `evals/metrics.py`, `evals/concurrency.py`, `evals/run_evals.py`, `evals/results/schema.py`, `docs/evaluation-metrics.md`, `localrag/cli/commands/eval.py` |
| Eval reproducibility (seed, run metadata) | `evals/environment.py`, `evals/run_evals.py`, [reproducibility.md](reproducibility.md) |
| Eval result versioning/comparison/baselines | `evals/results/`, `evals/compare.py`, `evals/baselines/`, `localrag/cli/commands/eval_compare.py` |
| Eval per-case failure analysis | `evals/failure_analysis.py`, `evals/results/schema.py`, `evals/report.py`, [ADR 031](adr/031-failure-analysis-contract.md) |
| Eval dataset schema / registry / fixtures | `evals/dataset/schema.py`, `evals/dataset/registry.py`, `evals/dataset/fixtures/`, [eval-datasets.md](eval-datasets.md) |
| Benchmark matrix contract / runner | `evals/matrix.py`, `localrag/cli/commands/benchmark.py`, [reproducibility.md](reproducibility.md) |
| Long-context live-local benchmark | `evals/long_context.py`, `localrag/cli/commands/benchmark.py`, [ADR 026](adr/026-long-context-benchmark-boundary.md) |
| Optional evaluation tracking | `evals/tracking.py`, `.env.example`, [ADR 018](adr/018-optional-mlflow-experiment-tracking.md) |
| Dockerized benchmark integration | `docker-compose.benchmark.yml`, `Dockerfile`, `scripts/docker_benchmark.py`, `docker/models.lock.json`, [ADR 033](adr/033-dockerized-benchmark-boundary.md) |
| Offline HTML benchmark reports | `evals/report.py`, `localrag/cli/commands/report.py`, README report usage |
| Benchmark leaderboard publication | `evals/leaderboard.py`, `localrag/cli/commands/leaderboard.py`, [benchmark-leaderboard.md](benchmark-leaderboard.md) |
| Late-interaction feasibility spike | `evals/late_interaction.py`, `research/70-late-interaction-spike/`, [ADR 027](adr/027-late-interaction-feasibility-boundary.md) |
| RAPTOR feasibility spike | `research/68-raptor-spike/`, `tests/test_raptor_spike.py`, [ADR 028](adr/028-raptor-feasibility-boundary.md); research-only, not default retrieval |
| GraphRAG feasibility spike | `research/67-graphrag-spike/`, `tests/test_graphrag_spike.py`, [ADR 029](adr/029-graphrag-feasibility-boundary.md); research-only, not default retrieval |
| Prompt / answer streaming | `localrag/rag/prompt.py`, `localrag/rag/engine.py` |
| Human Ollama install (not Python) | [ollama.md](ollama.md) |
| Human Tesseract install (not Python) | [ocr.md](ocr.md) |
| Contributor workflow wrappers | `Taskfile.yml`, [CONTRIBUTING](../.github/CONTRIBUTING.md) |

## Commands (uv)

```bash
uv sync --locked
uv run localrag --help
uv run pytest
uv run ruff format .
uv run ruff check .
```

Pre-commit and contribution workflow: [`.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md).
For portable task commands and their variable contract, use [`Taskfile.yml`](../Taskfile.yml).

## External dependencies

- **Ollama** runs outside the repo (CLI or Docker). LocalRAG talks over HTTP using `OLLAMA_BASE_URL` and model env vars.
- **Chroma** data is local filesystem under `CHROMA_PERSIST_PATH` (see `.env.example`).

## When to update this doc

Update [agent-navigation.md](agent-navigation.md) if you add major entry points, move packages, or change the “read order” anchors. Update [architecture.md](architecture.md) if layers, routers, schemas/services/repositories, or ingest/RAG flow change materially. See [AGENTS.md](../AGENTS.md) for the full “documentation maintenance for agents” rule.
