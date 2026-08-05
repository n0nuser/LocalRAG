# ADR 029: GraphRAG feasibility boundary

## Context

Graph extraction could help multi-hop or corpus-level questions, but it adds a
second index, local model calls, staleness, provenance obligations, and query
cost to the existing Chroma/BM25/RRF path. Production graph construction and
API integration are not part of issue #67.

## Decision

Keep the dependency-free prototype in `research/graphrag_spike/`. It is not
wired into ingestion, Chroma, API dependency injection, RAGAS, or the default
retriever. Graph mode is explicit and falls back to classic hits when graph
evidence is empty.

The schema is JSON v1. Entity identity is `casefold(NFKC(name))` plus normalized
type; aliases are not guessed. Relation identity is the directed tuple
`(source_entity_id, normalized_predicate, target_entity_id)`, so reverse edges
and contradictory predicates remain distinct. Every node and edge retains
source, chunk, citation, extractor, prompt version, and confidence provenance.
Duplicates merge provenance and retain the maximum confidence. Unknown relation
endpoints and malformed provider records are discarded or quarantined, never
silently converted into facts.

Extraction accepts only the strict Pydantic shape in `graphrag.py`, batches eight
chunks, caps input at 4,000 characters, retries twice, and quarantines malformed,
missing, mismatched, or oversized records. The fixture provider stands in for a
local Ollama structured-output call; no network or cloud call is made.

Artifacts record corpus, extractor, prompt/config, and schema identity. They are
named standalone JSON files, atomically replaced after fsync, and rejected on an
unknown schema. Source updates/deletes rebuild from the current chunk set via
`without_sources`; corpus, extractor, config, or schema changes invalidate the
artifact. Interrupted writes preserve the previous artifact. Graph files never
share the Chroma collection or lifecycle.

Retrieval expands an entity neighborhood with a maximum hop and neighbor bound,
supports source metadata filters, deduplicates citations, and enforces a context
budget. Composition is an opt-in additive step over caller-provided classic
dense/BM25/RRF results. Freshness, reranking, compression, and global/community
summaries are not claimed by this slice and must be specified before adoption.

## Evidence and adoption matrix

The fixture emits #73 matrix-manifest and #84 result-file-shaped JSON, with exact
dataset/version, fixture digest, model identifiers, config, hardware, repeated
warm query timings, build time, artifact bytes, Python allocation peak, and
failure count. It is deterministic contract evidence, not a quality benchmark.
RAGAS and manual-only evaluation remain unchanged. A real follow-up must use the
same #73/#84 dataset/model identifiers and report cold/warm latency, memory,
quality, failures, and variance against classic hybrid retrieval.

| Criterion | Adoption gate | Issue #67 evidence | Decision |
| --- | --- | --- | --- |
| Quality / multi-hop / global queries | NDCG@10 +5 points and reviewed citations on same corpus | No real model or annotated corpus | Defer |
| Cost / extraction | Token/time budget and rebuild cost recorded | Fixture only; no LLM tokens | Defer |
| Latency | Warm p95 <=2x classic; cold reported | Tiny fixture only | Defer |
| Storage / memory | Storage <=5x and RAM <=2x unless justified | Artifact and Python peak only | Defer |
| Dependency/license | Offline install and model terms reviewed | No new dependency; model follow-up pending | Defer |
| Freshness / maintenance | Incremental/delete/rebuild owner and migration plan | Standalone rebuild contract only | Defer |

**Decision: reject default adoption and defer.** A successful spike establishes
safe boundaries and a reproducible experiment, not a production graph feature.
