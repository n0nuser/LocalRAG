# ADR 036 — Retire zero-cost feature flags and unify the retrieval pipeline

- **Status:** accepted
- **Date:** 2026-08-05
- **Supersedes:** the opt-in halves of [ADR 022](022-context-compression-contract.md) and [ADR 024](024-embedding-cache-contract.md)
- **Issue:** [#143](https://github.com/n0nuser/LocalRAG/issues/143) (Finding 3)

## Context

`localrag/rag/` carried ten default-off `*_enabled` flags. Each added settings, a
branch in the retrieval path, tests, and an ADR. Because every flag defaulted to
`False`, the shipped default path exercised almost none of that code, and the
combinatorial space between flags was effectively untested.

Issue #143 proposed deciding per feature — promote, keep, or remove — "using the
existing benchmark evidence". No such evidence exists: all three runs in
`evals/results/` were recorded with every flag disabled, and one has null
metrics. A promote/remove verdict for the LLM-backed features would therefore
have been a guess about answer quality on a public configuration surface.

## Decision

Flags are retired **only where the feature costs nothing to leave on**. The test
is whether enabling the feature requires a provider round-trip or an optional
dependency.

**Retired — feature is now unconditional:**

| Flag | Why it was safe to retire |
| --- | --- |
| `context_compression_enabled` | `compress_contexts` is deterministic and extractive: regex-based sentence selection, no provider call, no optional dependency. |
| `embedding_cache_enabled` | `EmbeddingCache` is stdlib-only and bounded by `embedding_cache_max_entries` / `max_bytes`. |

**Kept — the flag gates a real cost or dependency:**

| Flag | Why the flag stays |
| --- | --- |
| `rerank_enabled` | Requires the `rerank` extra (`sentence_transformers`). Unconditional loading breaks any install without it. |
| `otel_enabled`, `langfuse_enabled` | Require optional exporters and a reachable collector. |
| `query_rewrite_enabled`, `query_expansion_enabled`, `hyde_enabled`, `adaptive_enabled`, `adaptive_critique_enabled` | Each adds one or more LLM round-trips per query; `adaptive` is a multi-round loop. Forcing these on would change default latency and cost. |

Retiring the two flags means their budget settings, not a boolean, decide
behavior. Context compression now always applies `context_compression_*` budgets
(default `max_contexts=5`, `total_tokens=1024`), and ingestion always writes to
`embedding_cache_path`.

Separately, `Retriever.retrieve` is split into ordered stages — `_plan_query`,
`_dense_retrieve`, `_fuse_candidates`, `_rerank`, post-processing — behind a
frozen `_QueryPlan`. This removes the `# noqa: C901, PLR0912, PLR0915` that the
previous single-function branch pile required.

## Consequences

- Default answers now use compressed context. This is the documented behavior of
  ADR 022, applied by default rather than opt-in.
- A default install writes an embedding cache directory it previously did not.
  It is bounded and can be relocated with `EMBEDDING_CACHE_PATH`, but it is a new
  on-disk side effect of a stock install.
- `EmbeddingCache(..., enabled=...)` is gone from the constructor.
- The retired names are **accepted and ignored** in YAML, `--set` overrides, and
  the environment, warning once with `DeprecationWarning`. No existing
  configuration fails to load.
- The promote/keep/remove decision for the eight remaining flags is deferred
  until per-flag benchmark runs exist. That work needs a benchmark matrix that
  varies one flag at a time, which the current harness supports but has not run.
