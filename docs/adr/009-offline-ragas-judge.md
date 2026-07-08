# ADR 009: RAGAS eval judge runs on local Ollama, not OpenAI

## Context

`evals/run_evals.py` gates the RAG pipeline's quality on faithfulness,
answer relevancy, context precision, and context recall (ragas metrics).
Ragas's `evaluate()` defaults its judge LLM to `OpenAI()`, which requires
`OPENAI_API_KEY` — not present anywhere in this project's `.env`, and at
odds with LocalRAG's offline-first positioning ("Your documents, your
models, your machine."). Installed `ragas==0.4.3` also deprecated the
`ragas.metrics.*` singleton import style this script originally used, in
favor of `ragas.metrics.collections.*` classes scored per-sample via async
`.ascore()`.

## Decision

Point the judge LLM and embeddings at the same local Ollama instance
LocalRAG itself uses, via Ollama's OpenAI-compatible `/v1` endpoint:

```python
ollama_client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")
judge_llm = llm_factory(judge_model, client=ollama_client, adapter="instructor")
judge_embeddings = OpenAIEmbeddings(client=ollama_client, model="nomic-embed-text")
```

`openai.AsyncOpenAI`/`ragas.embeddings.OpenAIEmbeddings` are used here purely
as clients for the OpenAI-shaped wire protocol Ollama speaks — `base_url`
points at localhost, the API key is a required-but-ignored dummy string, and
no request ever reaches OpenAI's actual service. This was chosen over:

1. **OpenAI as judge via CI secret** — rejected, can't be verified locally,
   contradicts offline-first positioning.
2. **LangChain wrapper** (`langchain-ollama` +
   `ragas.llms.LangchainLLMWrapper`) — works, but pulls in a dependency this
   project's owner prefers to avoid for what's fundamentally an HTTP call to
   a local model.
3. **Swap the whole eval framework to DeepEval** — rejected as a
   disproportionate framework migration for a compatibility fix; DeepEval
   isn't obviously lighter than ragas, just a different footprint.
4. **LiteLLM adapter** — avoids LangChain, but pulls in `litellm` +
   `instructor` for the same job the `openai` client (already a core
   dependency, since it backs the real `LLM_BACKEND=openai` provider) does
   natively.

Judge model defaults to `gemma3:4b`, embeddings to `nomic-embed-text` — the
project's own default LLM/embedder, already pulled by the existing Docker
Ollama container. Both are overridable via `--judge-model`/`--ollama-url`.

## Consequences

- Evals are fully offline-capable and reproducible on any machine running
  the project's own Docker stack — no external API key needed anywhere,
  including CI (`.github/workflows/evals-pr.yml` runs an `ollama/ollama`
  service container and pulls the judge/embedding models before scoring).
- Judge quality is bounded by a small local model (`gemma3:4b`, 4B params)
  rather than a GPT-4-class judge — `PASS_THRESHOLDS` may need recalibration
  if scores drift far from expectations on a real corpus. First full run:
  faithfulness 0.942, answer_relevancy 0.671, context_precision 0.891,
  context_recall 1.000 — all above threshold.
- Zero new dependencies added for this fix.

## Related

`docs/superpowers/plans/2026-07-07-production-rag-hardening.md` (Task 1),
session narrative in `log_reasonings/2026-07-07-ragas-offline-judge.md`.
