# ADR 024: Provider-Aware Ingestion Embedding Cache

## Status

Accepted; opt-in trigger superseded by [ADR 036](036-retire-zero-cost-feature-flags.md).

> **Amendment (ADR 036):** `EMBEDDING_CACHE_ENABLED` is retired and the cache is
> always active. The contract below is otherwise unchanged; `EMBEDDING_CACHE_PATH`
> and the entry/byte bounds still control it.

## Context

Re-ingesting unchanged chunks repeatedly invokes an embedding provider. A vector
cache can reduce that work, but a vector is only valid in the embedding space that
produced it. Model tags are mutable and source metadata is tenant- and path-specific,
so neither model names nor cached records may be treated as complete identity.

## Decision

LocalRAG caches **ingestion embeddings only** at the `EmbeddingProvider` seam.
Query embeddings are deliberately not cached by this change. `IngestionService`
continues to build source, tenant, chunk ID, and freshness metadata and performs the
same delete-then-upsert operation on a cache hit as on a provider result.

Each entry is a JSON record under `EMBEDDING_CACHE_PATH` (default
`./data/embedding-cache`) with only `schema`, hashed `key`, `dimension`, checksum,
and vector. The key is a SHA-256 digest of canonical JSON containing:

- scope (`ingestion`), provider identity, effective model, and provider revision;
- preprocessing version, task prefix, and requested/known vector dimension;
- SHA-256 of the UTF-8 chunk text.

The raw text, source path, tenant ID, secrets, and document metadata are never
persisted. A cache with an unknown dimension cannot hit until the provider reports
one; the returned vector dimension is validated before storage. This prevents a
vector from another embedding space being served. Model overrides are included in
the effective model field. Changing chunking or preprocessing settings changes the
preprocessing component; changed text, order, duplicates, source updates, and
deletes are handled by normal ingestion identity and upsert semantics rather than
cache metadata.

Writes use a same-directory temporary file followed by `os.replace`. Reads validate
schema, key, dimension, finite numeric values, and checksum; invalid entries are
deleted and recomputed. A process lock (`fcntl`) and re-entrant thread lock cover
lookup, provider fallback, writes, and eviction, so concurrent misses serialize and
never expose partial entries. Lock timeout, read-only/full storage, and other cache
I/O errors are fail-open: the provider result is returned and ingestion continues.
Entries are bounded by `EMBEDDING_CACHE_MAX_ENTRIES` and
`EMBEDDING_CACHE_MAX_BYTES`; least-recently-accessed JSON entries are evicted.
`EmbeddingCache.clear()` is the maintenance seam. The cache is disabled by default.

## Consequences

Warm ingestion can avoid provider calls for identical, compatible chunks. Cache
storage is local and bounded, but unknown provider revisions (for example a mutable
remote model tag that exposes no digest) are represented explicitly and do not claim
stronger invalidation than the provider reports. Cache statistics are available on
the cache object (`hits`, `misses`, `write_failures`) for callers and benchmarks.

The reproducible cold/warm benchmark is
`benchmarks/embedding_cache_benchmark.py`; it reports measurements and does not
assert that warm execution is universally faster. Evaluation workflows, including
RAGAS, remain manually invoked and are not coupled to this cache.
