# RAPTOR Feasibility Report

Issue: #68

## Decision

**Defer and reject default adoption.** The offline prototype validates a
versioned hierarchy, provenance-preserving summaries, atomic artifacts,
invalidation, failure recovery, bounded level retrieval, and source
deduplication. It does not establish that RAPTOR improves LocalRAG quality.

## Reproduction

```bash
uv run python research/68-raptor-spike/run_fixture.py
uv run pytest tests/test_raptor_spike.py
```

The runner writes `artifact.json`, `result.json`, and `manifest.json` under
`research/68-raptor-spike/artifacts/`. The fixture checksum, exact deterministic
embedding/summarizer identifiers, resolved config, Python/platform, build and
query timings, peak Python allocation, artifact size, and summary call count
are recorded. The result/manifest follow the existing #84/#73 evidence shape;
they are fixture evidence, not a RAGAS run.

## Parameters and contracts

| Area | Bounded prototype choice |
| --- | --- |
| Clustering | Seeded deterministic hash partition, two clusters, reduction factor 2; no learned k-means claim |
| Minimums | Minimum cluster size 2; singleton/undersized groups stay leaves |
| Recursion | Maximum 4 levels; stop when fewer than two minimum clusters or no reduction |
| Summary | Injectable function, versioned identity, 800-character cap; failed calls skipped |
| Context | Explicit level selection, weights `(1.0, .85, .7)`, 2,400-character cap |
| Identity | Stable node IDs, leaf content hashes, source and chunk IDs, embedding/config/model digest |
| Persistence | JSON schema v1, fsync and atomic replace, prior file preserved on failure |
| Mutation | Source update/delete rebuilds from leaves; incompatible identity invalidates |
| Retrieval | Level-aware scoring, source deduplication, deterministic ties, citation serialization |

Summary nodes and leaves have distinct `kind` values. Every summary carries all
descendant source IDs, chunk IDs, and leaf content hashes. Therefore retrieval
cannot return a summary citation without exact source references.

## Measurement boundary

The six-chunk fixture has no meaningful statistical power. It measures build
time, summary calls, serialized storage, Python allocation peak, and warm query
p50/p95. It does not measure LLM tokens because `fake-summary-v1` makes no LLM
call, nor cold process startup, RSS/VRAM, network, or real model quality. A
representative follow-up must run flat dense/BM25/hybrid and RAPTOR against the
same versioned #73/#84 dataset, with exact model revisions, resolved config,
hardware, cold/warm repetitions, NDCG/Recall and manual citation review. RAGAS
remains manual-only and unchanged.

## Unsupported cases

Real LLM summarization/token accounting, multilingual quality, prompt-injection
defense, learned clustering stability, online partial summary reuse, freshness
ordering, cross-level reranking/compression, native Chroma persistence, and
large-corpus memory behavior are unsupported. Summary vectors must not be
inserted into the existing Chroma collection without a separately named,
embedding-compatible space and an explicit production decision.

## Adoption criteria

Adopt only if quality improves NDCG@10 by at least 5 points over flat hybrid,
warm p95 is at most 2x flat, peak memory at most 2x, storage at most 5x unless
quality justifies it, citations remain complete, rebuild/staleness cost is
bounded, and maintenance/dependency ownership is explicit. Current evidence
passes contract tests only and fails the real-model evidence gates. No
production integration issue is opened by this spike.
