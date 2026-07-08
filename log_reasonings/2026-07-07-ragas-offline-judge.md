# 2026-07-07 — RAGAS judge LLM: stay offline, avoid new frameworks

## Context

Plan task "Gate RAGAS evals on PRs" hit a pre-existing break: `evals/run_evals.py`
imported legacy `ragas.metrics.collections` singletons; installed `ragas==0.4.3`
requires initialized `Metric` instances, and its per-metric result access
changed from scalar to per-row list. Separately, `ragas.evaluate()` defaults its
judge LLM to `OpenAI()`, which requires `OPENAI_API_KEY` — not present anywhere
in this project's `.env`, and fundamentally at odds with LocalRAG's
offline-first positioning ("Your documents, your models, your machine.").

## Options considered

1. **Keep OpenAI as judge**, rely on CI's `secrets.OPENAI_API_KEY`. Rejected —
   contradicts offline-first positioning; can't be verified locally at all.
2. **LangChain wrapper** (`langchain-ollama` + `ragas.llms.LangchainLLMWrapper`).
   Technically works, but user has a standing preference against pulling in
   LangChain for this kind of glue — sees it as heavy/abstraction-for-its-own-sake
   for what's fundamentally an HTTP call to a local model.
3. **Swap the whole eval framework to DeepEval** (suggested via pasted
   third-party/Gemini advice, with a `prometheus2` judge model). Rejected as
   disproportionate to the actual problem: this is a full eval-framework
   migration (new dataset shape, new metric API, new dependency, new judge
   model not even pulled yet), not a compatibility fix. DeepEval is not
   obviously "lighter" than ragas — it's a different framework with its own
   footprint, just not yet installed. Kept as a documented option for a future,
   separately-scoped task if ever revisited.
4. **Ollama via LiteLLM adapter** (`ragas.llms.LiteLLMStructuredLLM`). Avoids
   LangChain, but pulls in `litellm` + `instructor` for what amounts to talking
   OpenAI-wire-protocol to a local server that already speaks that protocol.
5. **Ollama via its native OpenAI-compatible `/v1` endpoint**, using the
   `openai` Python client already a core dependency of this project (LocalRAG
   already supports `LLM_BACKEND=openai` as a pluggable backend). Ragas ships
   `ragas.llms.llm_factory(...)` / `ragas.embeddings.OpenAIEmbeddings(...)`
   that accept any OpenAI-shaped client — pointing it at
   `http://localhost:11434/v1` with a dummy API key just works.

## Decision

Went with **option 5**. Zero new dependencies, no LangChain, no DeepEval
migration, no `prometheus2` download. `evals/run_evals.py` now builds an
`openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` client
and passes it into ragas's own `llm_factory`/`OpenAIEmbeddings` wrappers, judge
model defaulting to `gemma3:4b` (the project's own default LLM) and
`nomic-embed-text` for embeddings (the project's own default embedder) — both
already pulled by the existing Docker Ollama container.

## Consequences

- Evals are fully offline-capable and reproducible on any machine with the
  project's own Docker stack running — no external API key needed anywhere,
  including in CI (the `evals-pr.yml` gate no longer needs
  `secrets.OPENAI_API_KEY`).
- Judge quality is bounded by a small local model (`gemma3:4b`, 4B params)
  rather than GPT-4-class judges — thresholds in `PASS_THRESHOLDS` may need
  recalibration once a full real run completes; flagged as a follow-up if
  scores land far from expectations.
- `--judge-model`/`--ollama-url` CLI flags added so a different/larger local
  judge can be swapped in without code changes.

## Related

See `[[2026-07-07-agent-fanout-strategy]]` for how this fix was executed as
part of a larger multi-agent plan rollout, and `docs/adr/` for LocalRAG's
existing formal architecture decisions (this file is a lighter-weight,
session-level reasoning log rather than a settled ADR).
