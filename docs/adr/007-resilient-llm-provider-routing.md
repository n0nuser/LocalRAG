# ADR 007: Resilient, uniformly-routed LLM provider abstraction

## Context

`LLM_BACKEND` (ollama/openai/anthropic) existed as a setting, but `RAGEngine`
hand-rolled its own httpx calls to Ollama directly for prompt generation and
streaming, bypassing `localrag/llm/factory.py::build_provider` entirely.
Changing `LLM_BACKEND` had no effect on `/query` — it only ever hit Ollama.
Separately, no provider had retry, timeout backoff, or fallback behavior: a
transient Ollama hiccup or OpenAI/Anthropic rate limit surfaced immediately
as a user-facing failure.

## Decision

- Give `BaseLLMProvider` two abstract methods, `generate_from_prompt` and
  `stream_from_prompt`, implemented per backend (`ollama.py`,
  `openai_provider.py`, `anthropic_provider.py`).
- Wrap every provider `build_provider` returns in `ResilientProvider`
  (`localrag/llm/resilience.py`): tenacity retry with backoff, a pybreaker
  circuit breaker, and an optional configured fallback backend
  (`llm_fallback_backend`) that takes over once the breaker trips.
- Give `RAGEngine` a required `provider: BaseLLMProvider` field; route
  `_stream_chat_tokens` through `self.provider.stream_from_prompt(...)`
  instead of its own httpx client. `get_engine()` wires
  `provider=build_provider(settings)`.

## Consequences

- `LLM_BACKEND` now genuinely governs `/query` — verified live by setting
  `LLM_BACKEND=openai` with a bogus key and observing a real
  `401 AuthenticationError` from the OpenAI SDK instead of a silent
  fall-through to Ollama.
- All three backends get retry/circuit-breaker/fallback uniformly, instead of
  resilience logic living only where someone happened to add it.
- Every LLM-calling code path (RAG answers, ragas eval judge, query
  rewriting) now goes through one seam (`build_provider`), so future
  resilience or observability changes apply everywhere at once.

## Related

`[[../architecture.md]]`, `docs/superpowers/plans/2026-07-07-production-rag-hardening.md`
(Tasks 13–14).
