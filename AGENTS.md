# Agent and contributor context

Hermit keeps **human-oriented** docs in the [README](README.md) and **machine- and agent-oriented** maps under [`docs/`](docs/). Use these to load the right files first and avoid spelunking the whole tree.

## Trunk-based Git (read this before branching)

`main` is the only long-lived branch. **Do not** keep a personal or team **`develop`** for routine work—it slows integration and fights trunk-based development. Branch short-lived **`feat/…`** or **`fix/…`** from an **updated `main`**, open PRs **to `main`**, and integrate with **`git rebase origin/main`** (never merge commits for that). On GitHub use **Rebase** or **Squash** merge only. Full policy: [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md).

**Repeating workflow** (confirm branch merged if not `main`, checkout `main`, pull, stash pop if needed, new `feat/…`): follow [`.cursor/skills/trunk-feature-workflow/SKILL.md`](.cursor/skills/trunk-feature-workflow/SKILL.md).

| Document | What it explains | Update when |
| --- | --- | --- |
| [docs/agent-navigation.md](docs/agent-navigation.md) | Efficient context loading: read order, “if you change X open Y”, uv commands, pointers to `.cursor/rules` and CONTRIBUTING | Entry points, toolchain, or navigation hints change |
| [docs/architecture.md](docs/architecture.md) | Package layers, ingest/query data flow, extension points (new parser, router, CLI command, setting) | Package layout, routers, ingestion/RAG pipeline, or DI wiring changes |
| [docs/ollama.md](docs/ollama.md) | Installing and running Ollama (host vs Docker), default models, links to upstream docs | Default models in `.env.example` / `Settings`, or Ollama-related workflows change |
| [`.cursor/skills/trunk-feature-workflow/SKILL.md`](.cursor/skills/trunk-feature-workflow/SKILL.md) | Trunk Git steps: merged check (when not on `main`), `main` + pull, stash/unstash around checkout/pull, new `feat/…` | This skill’s steps or CONTRIBUTING trunk rules change |

**Maintenance:** When you change behavior or structure covered by a row above, update the corresponding doc in the same PR whenever the drift would confuse the next reader (human or agent).
