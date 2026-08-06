# Agent and contributor context

LocalRAG keeps **human-oriented** docs in the [README](README.md) and **machine- and agent-oriented** maps under [`docs/`](docs/). Use these to load the right files first and avoid spelunking the whole tree.

**Start here:** [`docs/agent-navigation.md`](docs/agent-navigation.md) is the maintained "I'm changing X — open Y" table and read order. It is more detailed than this file and is kept current; consult it before opening modules. This file covers the rules that table does not: Git policy, layering contracts, and what you must update when you change things.

## Toolchain

Python **3.13+**, dependencies and commands through **uv**. [`Taskfile.yml`](Taskfile.yml) is the portable wrapper contract (`task install`, `task test`, `task lint`, `task format`, `task ingest`, `task benchmark`, `task docker-up`, …); `task --list` enumerates it. Installing Task, uv, Docker, and Ollama is covered in [CONTRIBUTING](CONTRIBUTING.md#prerequisites-and-installation); every task also has a plain `uv run …` equivalent, so Task is optional.

```bash
uv sync --locked
uv run localrag --help
uv run pytest -m "not integration"
uv run ruff check . && uv run ruff format --check .
uv run mypy localrag/ --ignore-missing-imports --no-strict-optional
uv run bandit -r localrag/ -ll
```

Gates enforced by [`.pre-commit-config.yaml`](.pre-commit-config.yaml): ruff (lint + format), bandit, mypy, unit pytest on commit, integration tests on **pre-push** (`scripts/run_integration_tests.py`, needs Docker), and Conventional Commits on `commit-msg`. Coverage `fail_under = 80` over `localrag/` (CLI modules are omitted deliberately — they are thin wrappers tested through services). Ruff runs `select = ["ALL"]` with an explicit ignore list in `pyproject.toml`; do not widen that list to silence a finding you should fix.

Tests marked `integration` require a running stack and are excluded from the default run.

**Unit tests run locally (`task test`); end-to-end tests run in Docker (`task test-integration`) — always.** Rebuild the image before trusting an end-to-end result: a stack that has been up since before your changes is still serving the old artifact, dependencies included. The unit suite fakes the vector store, so a green run is **not** evidence that storage or retrieval works — anything that can only fail against real Chroma (batch limits, `include` contracts, parser routing, embedding quality) needs an `integration` test. Read [`.cursor/rules/testing.mdc`](.cursor/rules/testing.mdc) before writing tests: it covers the container's constraints (read-only rootfs, no pytest/pip) and why retrieval tests must go through `get_retriever()` rather than Chroma's `query_texts`.

## Trunk-based Git (read this before branching)

`main` is the only long-lived branch. **Do not** keep a personal or team **`develop`** for routine work—it slows integration and fights trunk-based development. Branch short-lived **`feat/…`** or **`fix/…`** from an **updated `main`**, open PRs **to `main`**, and integrate with **`git rebase origin/main`** (never merge commits for that). On GitHub use **Rebase** or **Squash** merge only. Full policy: [`CONTRIBUTING.md`](CONTRIBUTING.md).

**Repeating workflow** (confirm branch merged if not on `main`, checkout `main`, pull, stash pop if needed, new `feat/…`): follow [`.cursor/skills/trunk-feature-workflow/SKILL.md`](.cursor/skills/trunk-feature-workflow/SKILL.md).

## Coding rules live in `.cursor/rules/`

Non-obvious Python constraints are **not** duplicated here. Read them before editing Python:

| File | Scope |
| --- | --- |
| [`.cursor/rules/documentation-maintenance.mdc`](.cursor/rules/documentation-maintenance.mdc) | **Mandatory.** Docs ship in the same commit as the code. Which document each kind of change obliges you to update, and how to write them |
| [`.cursor/rules/change-discipline.mdc`](.cursor/rules/change-discipline.mdc) | **Mandatory.** Verify before claiming; never silence a diagnostic; delete workarounds with their cause; no cwd-dependent paths; importable names; stay in scope |
| [`.cursor/rules/critical-rules.mdc`](.cursor/rules/critical-rules.mdc) | Non-negotiable: no logic in `__init__` (use `@classmethod` factories), imports top-level only, no `__all__` outside `__init__.py`, no `Field()` in non-FastAPI models, no isolated model/`__init__` tests |
| [`.cursor/rules/python-conventions.mdc`](.cursor/rules/python-conventions.mdc) | SOLID + YAGNI, dependency injection, snake_case Pydantic fields with `alias_generator`, specific exceptions, business-decision comments |
| [`.cursor/rules/testing.mdc`](.cursor/rules/testing.mdc) | Test layout and expectations |
| [`.cursor/rules/code-review.mdc`](.cursor/rules/code-review.mdc), [`.cursor/rules/grug.mdc`](.cursor/rules/grug.mdc), [`.cursor/rules/agent-communication.mdc`](.cursor/rules/agent-communication.mdc) | Review posture, simplicity preference, agent reporting style |

## Package map

`localrag/` is the application; `evals/` is the evaluation and benchmarking harness; both ship in the repo and both have contracts recorded as ADRs.

| Package | Role |
| --- | --- |
| `localrag/application/` | Transport-neutral use cases, DTOs, errors, jobs, repositories, and runtime container |
| `localrag/api/` | FastAPI HTTP adapter — see the DDD split below |
| `localrag/mcp/` | MCP adapter (FastMCP SDK) over stdio and HTTP |
| `localrag/cli/` | Typer app (`localrag.cli.app:app`); one module per command in `cli/commands/` |
| `localrag/ingestion/` | Loader, parsers (`anydoc`, `pdf`, `docx`, `markdown`, `code`, `text`), chunking (`contract`, `structural_chunker`, `chunker`, `recursive_chunker`), embedder, `service` |
| `localrag/rag/` | `engine`, `retriever`, `bm25_index`, `reranker`, `adaptive`, `compressor`, `hyde`, `query_rewrite`, `query_cache`, `prompt` |
| `localrag/embedding/` | Provider abstraction (`base`, `factory`, `sentence_transformers`) and ingestion `cache` |
| `localrag/llm/` | `factory`, `providers/` (ollama, openai, anthropic), `resilience`, `costs`, `types` |
| `localrag/storage/` | `vector_store` — Chroma persistence boundary |
| `localrag/plugins/` | Versioned third-party retriever contract (`retriever.py`) |
| `localrag/observability/` | Optional OpenTelemetry `tracing` |
| `localrag/agent/` | Tool-use agent service (`search_documents` / `answer_directly`) |
| `localrag/ollama/` | Ollama HTTP request/response schemas shared by embedder, engine, health, setup |
| `localrag/` root | `settings.py`, `logging_config.py`, `metrics.py`, `audit.py` |
| `evals/` | `run_evals`, `metrics`, `matrix`, `compare`, `leaderboard`, `report`, `failure_analysis`, `concurrency`, `environment`, `tracking`, `long_context`, `late_interaction`, `dataset/`, `results/`, `baselines/` |

`evals/` **ships in the wheel** alongside `localrag/` (`[tool.hatch.build.targets.wheel]`), because `localrag/cli/commands/` imports it for the `benchmark`, `report`, `leaderboard`, `eval`, and `eval-compare` commands. It is linted like the rest of the package; the few inherent exemptions are scoped in `[tool.ruff.lint.per-file-ignores]`. The `eval` adapters spawn it with `python -m evals.…` rather than a `__file__`-relative path, which would not resolve once installed — `tests/test_packaging.py` guards all of this.

## Basic DDD layout (HTTP API)

The FastAPI layer follows a **light domain-driven** split:

| Piece | Location | Role |
| --- | --- | --- |
| **Schemas** (request/response DTOs, OpenAPI) | `localrag/api/schemas.py` | Pydantic models and path type aliases only—no business rules. |
| **Application services** | `localrag/application/` | Transport-neutral use cases: orchestration, validation, logging, and domain errors. |
| **Repositories** | `localrag/application/repository.py` | Persistence boundaries for application use cases (e.g. Chroma collections via `VectorStore`). |
| **HTTP adapters** | `localrag/api/routers/*.py`, `localrag/api/service.py` | Routes and schema/error mapping: dependencies, call application services, return HTTP responses. **No** domain logic in adapter modules. |
| **Dependency injection** | `localrag/api/dependencies.py` | Shared service instances, `require_api_key`. |
| **Background jobs** | `localrag/application/jobs.py` | Async ingest job registry behind `ingest_directory_async` / `get_ingest_job`. |
| **Middleware** | `localrag/api/middleware.py` | Request ID, logging. |
| **MCP adapter** | `localrag/mcp/` | FastMCP tool registration and transport adapters over the same application use cases. |
| **Cross-cutting API errors** | `localrag/api/exceptions.py` | `HttpMappedError` subclasses (`IngestApiError`, `RagApiError`) mapped to HTTP in `localrag/api/main.py`. |

Routers: `health`, `ingest`, `query`, `collections`, `agent`, `metrics`.

Domain packages (`localrag/ingestion/`, `localrag/rag/`, `localrag/storage/`) keep their own services and types; the API service calls into them (e.g. `IngestionService`, `RAGEngine`).

## Configuration

Everything resolves into `Settings` (`localrag/settings.py`) via `load_settings` / `get_settings` / `set_current_settings`. `settings_customise_sources` fixes the precedence, **highest first**:

1. `--set FIELD=VALUE` CLI overrides (passed as init values; unknown fields raise `ConfigError`)
2. environment variables
3. `.env`
4. structured YAML from `--config` / `LOCALRAG_CONFIG` (see [`config.example.yaml`](config.example.yaml))
5. file secrets

So YAML is a **base layer that env and CLI override**, not the other way around. [`.env.example`](.env.example) is the canonical env var list and must stay in sync with `Settings`. Contract: [ADR 020](docs/adr/020-structured-configuration.md).

`Settings` is **grouped** into per-feature sub-models ([`localrag/settings_groups.py`](localrag/settings_groups.py)), each owning its own validation, while the documented **flat** names stay the public surface via [`localrag/settings_map.py`](localrag/settings_map.py) and generated properties. Adding a setting means touching all three: the group model, `FLAT_TO_PATH`, and a flat property — a totality test fails if you miss one. Use `settings.with_overrides(...)`, never `model_copy(update=...)`, for flat names. Contract: [ADR 037](docs/adr/037-grouped-configuration-model.md).

## Roadmap and ADRs

- [`ROADMAP.md`](ROADMAP.md) is a **contributor-facing summary, not a second issue tracker** — GitHub owns live issue and milestone state. It is machine-checked by [`scripts/validate_roadmap.py`](scripts/validate_roadmap.py) against live GitHub data (or a JSON fixture), which requires every live milestone to be linked and headed exactly once. Run it after editing the roadmap.
- Record durable architectural decisions in [`docs/adr/`](docs/adr/); [`docs/adr/README.md`](docs/adr/README.md) indexes every ADR by status (accepted / amended / research boundary). Per CONTRIBUTING, public extension or trust boundaries and behavior-changing defaults **require** an ADR. There are 37 today; new contracts continue the numbering. Boundary-setting research spikes (late interaction, RAPTOR, GraphRAG) are recorded as ADRs too, and stay research-only rather than becoming default retrieval.

## Claude Code skills for this repo

If you're running as Claude Code, these installed skills/subagents map directly onto LocalRAG's stack—reach for them instead of re-deriving generic advice:

| Area | Skill / agent | When it applies here |
| --- | --- | --- |
| Git workflow | `superpowers:using-git-worktrees`, `git-pr-workflows:git-workflow` | Isolating longer-lived `feat/…`/`fix/…` work, or orchestrating review → PR beyond the basic steps in `.cursor/skills/trunk-feature-workflow/SKILL.md` |
| API/DDD layering | `backend-development:architecture-patterns`, `backend-development:api-design-principles` | Reshaping the schemas/service/repository/router split above, or designing new routes |
| Python style & types | `python-development:python-code-style`, `python-development:python-type-safety` | Anything ruff/mypy already gate in `.pre-commit-config.yaml`—use before relying on defaults |
| Config & settings | `python-development:python-configuration` | Changes to `localrag/settings.py` / `.env.example` / `config.example.yaml` (pydantic-settings) |
| Error handling & resilience | `python-development:python-error-handling`, `python-development:python-resilience` | `HttpMappedError` subclasses, `localrag/llm/resilience.py`, ingestion retry/batch logic |
| Observability | `python-development:python-observability` | structlog / Prometheus / OTel work (`localrag/logging_config.py`, `localrag/metrics.py`, `localrag/observability/tracing.py`, `api/routers/metrics.py`) |
| Testing | `python-development:python-testing-patterns`, `superpowers:test-driven-development` | Adding/extending `tests/` (pytest, pytest-asyncio, respx) |
| Debugging | `superpowers:systematic-debugging`, `diagnosing-bugs` | Any bug/regression—before proposing a fix |
| LLM/provider code | `claude-api` | Anthropic/OpenAI/Ollama provider work in `localrag/llm/` |
| Security | `security-review`, `backend-api-security:backend-security-coder` | Auth (`API_KEY`), path validation (`is_path_allowed`), upload handling, anything bandit flags |
| Verification | `verify` | Before claiming a change works—exercise the flow, don't just trust lint/tests |

This is a pointer, not a guarantee of installation—confirm availability in your environment before relying on one.

## Documentation maintenance for agents

This is **mandatory and enforced by [`.cursor/rules/documentation-maintenance.mdc`](.cursor/rules/documentation-maintenance.mdc)**, which applies to every agent working in this repo. The table below is the authoritative version of that rule's trigger list.

When you change anything that affects **how agents find or reason about the codebase**, update the relevant docs **in the same change** (same PR). At minimum:

- **[docs/agent-navigation.md](docs/agent-navigation.md)** — new entry points, moved paths, or new “if you change X open Y” rows.
- **[docs/architecture.md](docs/architecture.md)** — layers, data flow, DI, or extension points that shifted.
- **[docs/adr/](docs/adr/)** — a new ADR when you set or move a public extension boundary, trust boundary, or behavior-changing default.
- **Other rows in the table below** — if the listed “Update when” condition applies.

Do not rely on agents discovering structural changes from code alone; keep the maps truthful.

| Document | What it explains | Update when |
| --- | --- | --- |
| [docs/agent-navigation.md](docs/agent-navigation.md) | Efficient context loading: read order, “if you change X open Y”, uv commands, pointers to `.cursor/rules` and CONTRIBUTING | Entry points, toolchain, navigation hints, or API layer layout change |
| [docs/architecture.md](docs/architecture.md) | Package layers, ingest/query data flow, extension points (new parser, router, CLI command, setting) | Package layout, routers, schemas/services/repositories, ingestion/RAG pipeline, or DI wiring changes |
| [docs/configuration.md](docs/configuration.md) | Every setting, default, and the resolution order | `Settings` fields, defaults, or the YAML schema change — keep `.env.example` in sync too |
| [docs/document-formats.md](docs/document-formats.md) | Supported file types and which parser handles each | `loader.py` extension sets or parser routing change |
| [docs/cli.md](docs/cli.md) | CLI command surface and options | Commands added/removed in `localrag/cli/commands/` or their flags change |
| [docs/rag-retrieval.md](docs/rag-retrieval.md) | Retrieval modes, hybrid ranking, reranking, compression, adaptive policy | Retrieval behavior or its settings change |
| [docs/plugin-author-guide.md](docs/plugin-author-guide.md) | Third-party retriever plugin contract and discovery | `localrag/plugins/retriever.py`, the contract version, or `examples/retriever-plugin/` change |
| [docs/observability.md](docs/observability.md) | structlog, Prometheus metrics, optional OTel tracing | Logging/metrics/tracing surface or `OTEL_*` settings change |
| [docs/deployment.md](docs/deployment.md) | Kubernetes persistence, dependency, probe, and security contract | `k8s/`, Compose files, or the deployment security contract change |
| [docs/data-lifecycle.md](docs/data-lifecycle.md) | Upload and query-audit retention | Upload/audit lifecycle settings or cleanup behavior change |
| [docs/ocr.md](docs/ocr.md) | Tesseract install and PDF OCR fallback | `OCR_*` settings or `parsers/pdf.py` OCR behavior change |
| [docs/ollama.md](docs/ollama.md) | Installing and running Ollama (host vs Docker), default models, links to upstream docs | Default models in `.env.example` / `Settings`, or Ollama-related workflows change |
| [docs/evaluation.md](docs/evaluation.md) | Evaluation entry point: pipeline order, commands, and which detail page owns what | The evaluation pipeline gains or loses a stage, command, or contract |
| [docs/reproducibility.md](docs/reproducibility.md) | Seeding, run metadata captured into `evals/results/*.json`, and known non-determinism limits | Eval seeding/sampling, `SNAPSHOT_SETTINGS_FIELDS`, generation sampling settings, or the results JSON shape change |
| [docs/eval-datasets.md](docs/eval-datasets.md) | Dataset manifest schema, registry, fixture authoring, offline-artifact rules | `evals/dataset/schema.py` field meaning, registry behavior, or bundled fixtures change |
| [docs/evaluation-metrics.md](docs/evaluation-metrics.md) | Metric contract and bounded RAGAS scoring | `evals/metrics.py` or the metric contract change |
| [docs/evaluation-reports.md](docs/evaluation-reports.md) | Offline HTML benchmark reports | `evals/report.py` or report output change |
| [docs/benchmark-leaderboard.md](docs/benchmark-leaderboard.md) | Published leaderboard and its strictness rules | `evals/leaderboard.py` or publication criteria change |
| [ROADMAP.md](ROADMAP.md) | Milestone/phase summary mirroring GitHub | Milestones or issue status change — then run `scripts/validate_roadmap.py` |
| [`.cursor/skills/trunk-feature-workflow/SKILL.md`](.cursor/skills/trunk-feature-workflow/SKILL.md) | Trunk Git steps: merged check (when not on `main`), `main` + pull, stash/unstash around checkout/pull, new `feat/…` | This skill’s steps or CONTRIBUTING trunk rules change |

**Maintenance:** When you change behavior or structure covered by a row above, update the corresponding doc in the same PR whenever the drift would confuse the next reader (human or agent).
