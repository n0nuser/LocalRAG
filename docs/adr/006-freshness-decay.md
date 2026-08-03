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

## Amendment: recency as a fusion signal in hybrid mode

The original decision multiplied the fused score by the decay factor. That is
sound when scores are spread out, but hybrid retrieval fuses with Reciprocal Rank
Fusion, whose scores are deliberately compressed — they encode rank position, not
relevance magnitude. At the default `rrf_k=60` the entire top-20 spans **1.31x**,
and adjacent ranks differ by **1.64%**, while the decay factor spans **4598x**
over a year.

Multiplying one by the other let recency dominate. Measured against the shipped
defaults:

- A rank-1 result lost to a fresh rank-2 result after **0.70 days**.
- It lost to a fresh rank-5 result after **2.75 days**.
- Decay was applied *after* the cross-encoder reranker, discarding its ordering.

In effect, hybrid retrieval ranked by ingestion date. The problem was the
mechanism, not the goal: "older content should not outrank current policy text"
remains right.

### Revised decision

Recency contributes as **its own ranked list inside the fusion**, which is what a
rank-based algorithm is designed for — RRF combines heterogeneous relevance
indicators without requiring them to share a scale
([Elasticsearch RRF reference](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)).

- Candidates are ranked newest-first; that rank contributes
  `freshness_weight / (rrf_k + rank)`, with `freshness_weight` (default `0.15`)
  taken out of the relevance budget so the weights still sum to 1.
- Relevance ties are broken by recency *before* ranking, so equally-relevant
  candidates are no longer ordered by sort stability. This is what preserves the
  original intent: when two chunks match equally well, the newer one wins.
- Candidates with no usable `ingested_at` take the **middle** recency rank.
  Dropping them would forfeit the recency weight, which is itself a penalty for
  missing metadata.
- `apply_freshness` no longer rescores in hybrid mode; it still populates
  `freshness_factor` for observability. Vector-only mode keeps the multiplicative
  decay, where scores spread ~3x and it behaves as the intended tiebreaker.
- Both opt-outs still work: `freshness_half_life_days=0` or
  `freshness_weight=0.0` collapses fusion to relevance only.

### Consequences of the amendment

- A clearly better match is no longer buried by age; recency only decides
  near-ties.
- `freshness_weight` is a new tunable, and the effective relevance weight is now
  `1 - freshness_weight`, so a large value trades away relevance.
- The value `0.15` is reasoned from the RRF gap arithmetic above, not tuned
  against a labelled corpus. Worth revisiting with the RAGAS eval.
- `ingested_at` remains ingestion wall-clock time, so a bulk re-ingest still
  flattens the recency signal. `source_mtime` is already captured in the same
  metadata and is the better input — left as follow-up.
