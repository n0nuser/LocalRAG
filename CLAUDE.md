# Agent and contributor context

LocalRAG keeps **human-oriented** docs in the [README](README.md) and **machine- and agent-oriented** maps under [`docs/`](docs/). Use these to load the right files first and avoid spelunking the whole tree.

## Trunk-based Git (read this before branching)

`main` is the only long-lived branch. **Do not** keep a personal or team **`develop`** for routine work—it slows integration and fights trunk-based development. Branch short-lived **`feat/…`** or **`fix/…`** from an **updated `main`**, open PRs **to `main`**, and integrate with **`git rebase origin/main`** (never merge commits for that). On GitHub use **Rebase** or **Squash** merge only. Full policy: [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md).

**Repeating workflow** (confirm branch merged if not on `main`, checkout `main`, pull, stash pop if needed, new `feat/…`): follow [`.cursor/skills/trunk-feature-workflow/SKILL.md`](.cursor/skills/trunk-feature-workflow/SKILL.md).

## Basic DDD layout (HTTP API)

The FastAPI layer follows a **light domain-driven** split:

| Piece | Location | Role |
| --- | --- | --- |
| **Schemas** (request/response DTOs, OpenAPI) | `localrag/api/schemas.py` | Pydantic models and path type aliases only—no business rules. |
| **Application services** | `localrag/api/service.py` | Use cases: orchestration, validation, logging, mapping to/from schemas. |
| **Repositories** | `localrag/api/repository.py` | Persistence boundaries for the API (e.g. Chroma collections via `VectorStore`). |
| **HTTP adapters** | `localrag/api/routers/*.py` | Routes: dependencies, call services, return responses. **No** Pydantic models or domain logic in router modules. |
| **Cross-cutting API errors** | `localrag/api/exceptions.py` | `HttpMappedError` subclasses (`IngestApiError`, `RagApiError`) mapped to HTTP in `localrag/api/main.py`. |

Domain packages (`localrag/ingestion/`, `localrag/rag/`, `localrag/storage/`) keep their own services and types; the API service calls into them (e.g. `IngestionService`, `RAGEngine`).

## Claude Code skills for this repo

If you're running as Claude Code, these installed skills/subagents map directly onto LocalRAG's stack—reach for them instead of re-deriving generic advice:

| Area | Skill / agent | When it applies here |
| --- | --- | --- |
| Git workflow | `superpowers:using-git-worktrees`, `git-pr-workflows:git-workflow` | Isolating longer-lived `feat/…`/`fix/…` work, or orchestrating review → PR beyond the basic steps in `.cursor/skills/trunk-feature-workflow/SKILL.md` |
| API/DDD layering | `backend-development:architecture-patterns`, `backend-development:api-design-principles` | Reshaping the schemas/service/repository/router split above, or designing new routes |
| Python style & types | `python-development:python-code-style`, `python-development:python-type-safety` | Anything ruff/mypy already gate in `.pre-commit-config.yaml`—use before relying on defaults |
| Config & settings | `python-development:python-configuration` | Changes to `localrag/settings.py` / `.env.example` (pydantic-settings) |
| Error handling & resilience | `python-development:python-error-handling`, `python-development:python-resilience` | `HttpMappedError` subclasses, ingestion retry/batch logic |
| Observability | `python-development:python-observability` | structlog / Prometheus metrics work (`localrag/logging_config.py`, `api/routers/metrics.py`) |
| Testing | `python-development:python-testing-patterns`, `superpowers:test-driven-development` | Adding/extending `tests/` (pytest, pytest-asyncio, respx) |
| Debugging | `superpowers:systematic-debugging`, `diagnosing-bugs` | Any bug/regression—before proposing a fix |
| LLM/provider code | `claude-api` | Anthropic/OpenAI/Ollama provider work in `localrag/llm/` |
| Security | `security-review`, `backend-api-security:backend-security-coder` | Auth (`API_KEY`), path validation (`is_path_allowed`), upload handling, anything bandit flags |
| Verification | `verify` | Before claiming a change works—exercise the flow, don't just trust lint/tests |

This is a pointer, not a guarantee of installation—confirm availability in your environment before relying on one.

## Documentation maintenance for agents

When you change anything that affects **how agents find or reason about the codebase**, update the relevant docs **in the same change** (same PR). At minimum:

- **[docs/agent-navigation.md](docs/agent-navigation.md)** — new entry points, moved paths, or new “if you change X open Y” rows.
- **[docs/architecture.md](docs/architecture.md)** — layers, data flow, DI, or extension points that shifted.
- **Other rows in the table below** — if the listed “Update when” condition applies.

Do not rely on agents discovering structural changes from code alone; keep the maps truthful.

| Document | What it explains | Update when |
| --- | --- | --- |
| [docs/agent-navigation.md](docs/agent-navigation.md) | Efficient context loading: read order, “if you change X open Y”, uv commands, pointers to `.cursor/rules` and CONTRIBUTING | Entry points, toolchain, navigation hints, or API layer layout change |
| [docs/architecture.md](docs/architecture.md) | Package layers, ingest/query data flow, extension points (new parser, router, CLI command, setting) | Package layout, routers, schemas/services/repositories, ingestion/RAG pipeline, or DI wiring changes |
| [docs/ollama.md](docs/ollama.md) | Installing and running Ollama (host vs Docker), default models, links to upstream docs | Default models in `.env.example` / `Settings`, or Ollama-related workflows change |
| [`.cursor/skills/trunk-feature-workflow/SKILL.md`](.cursor/skills/trunk-feature-workflow/SKILL.md) | Trunk Git steps: merged check (when not on `main`), `main` + pull, stash/unstash around checkout/pull, new `feat/…` | This skill’s steps or CONTRIBUTING trunk rules change |

**Maintenance:** When you change behavior or structure covered by a row above, update the corresponding doc in the same PR whenever the drift would confuse the next reader (human or agent).
