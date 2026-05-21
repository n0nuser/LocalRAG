# ADR 006: Freshness-aware retrieval scoring

## Context

Chunk metadata already stores `ingested_at`, but ranking ignored recency. Older
content could outrank current policy text purely by lexical or semantic match.

## Decision

Apply exponential freshness decay in the retriever after ranking/fusion:

- `freshness_factor = 0.5 ** (age_days / freshness_half_life_days)`
- Multiply candidate score by the factor when valid `ingested_at` is present.
- Keep behavior opt-out with `freshness_half_life_days=0`.
- Surface `freshness_factor` in retrieval contexts for observability.

## Consequences

- Reduces stale-answer risk with minimal runtime overhead.
- Ranking becomes time-sensitive and dependent on metadata quality.
- Misformatted timestamps gracefully fall back to no decay.
