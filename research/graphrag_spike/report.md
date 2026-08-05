# GraphRAG Feasibility Report

Issue: #67

## Decision

**Reject default adoption and defer.** The prototype proves strict extraction
failure handling, deterministic graph identity, provenance, atomic versioned
artifacts, explicit mutation/invalidation, bounded filtered traversal, and
classic fallback. It does not prove GraphRAG quality, multi-hop/global-query
benefit, or acceptable real-model cost.

## Reproduction

```bash
uv run python research/graphrag_spike/run_fixture.py
uv run pytest tests/test_graphrag_spike.py
```

The runner writes `artifact.json`, `result.json`, and `manifest.json` below
`research/graphrag_spike/artifacts/`. The files are local run output, not
production Chroma data. The fixture uses `localrag-graphrag-fixture` version
`1.0.0`, `fixture-structured-extractor-v1`, and
`fixture-hash-embedding-v1`; the report must not be read as a live Ollama result.

## Boundaries

Graph facts are only accepted from the versioned structured schema. Batches are
bounded to eight inputs and 4,000 characters per chunk; malformed output is
retried twice and then quarantined. Missing files, oversized inputs, unknown
endpoints, empty/disconnected graphs, stale schema, and interrupted writes have
explicit safe behavior. Nodes and directed edges preserve source/chunk/citation
provenance and extraction metadata. Graph retrieval is bounded to two hops and
20 neighbors, filterable by source, citation-deduplicated, and context-budgeted.

Classic dense/BM25/RRF retrieval remains the caller's baseline and fallback.
Graph mode is opt-in. Freshness ordering, reranking, context compression,
community/global summaries, and Chroma compatibility are intentionally not
implemented.

## Measurement boundary

The fixture records build time, artifact bytes, Python allocation peak, repeated
warm query p50/p95, selected IDs, exact model/dataset/config/hardware provenance,
and extraction failures in #73/#84-shaped artifacts. It cannot establish quality,
token cost, RSS/VRAM, cold-start latency, corpus-scale memory, multilingual or
PII behavior, prompt-injection robustness, licensing of a chosen model, or
maintenance cost. A real comparison must use the canonical #73/#84 contracts,
exact dataset/model digests, cold and warm repetitions, classic hybrid baseline,
NDCG/recall and manual citation review. RAGAS remains manual-only.

## Adopt/reject matrix

| Area | Required evidence | Current result |
| --- | --- | --- |
| Quality and multi-hop/global benefit | Same annotated corpus, +5 NDCG@10, citation review | Not measured; defer |
| Extraction/index cost | Real local model tokens, time, failures, rebuild/update cost | Fixture only; defer |
| Latency | Warm p95 <=2x classic plus cold run | Fixture only; defer |
| Storage and memory | Artifact size <=5x and RAM <=2x unless quality justifies | Fixture numbers are not scalable; defer |
| Dependency/license | Offline locked environment and model/license inventory | No new dependency; model undecided |
| Maintenance/freshness | Owner, migration, invalidation and lifecycle plan | Standalone artifact only; defer |

No production integration, plugin architecture, default change, or follow-up
adoption claim is made by this slice.
