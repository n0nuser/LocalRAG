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
| **Cross-cutting API errors** | `localrag/api/exceptions.py` | Exceptions mapped to HTTP in `localrag/api/main.py` (e.g. `IngestApiError`). |

Domain packages (`localrag/ingestion/`, `localrag/rag/`, `localrag/storage/`) keep their own services and types; the API service calls into them (e.g. `IngestionService`, `RAGEngine`).

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
