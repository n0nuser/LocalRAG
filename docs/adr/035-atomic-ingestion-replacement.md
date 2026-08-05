# ADR 035: Serialized atomic source replacement

## Context

Ingestion replaces all chunks belonging to a source. Chroma does not expose a
transaction spanning the upsert and obsolete-vector delete, so concurrent
replacements could interleave and a failed write could remove the last good
version. BM25 also needs to avoid exposing a half-built corpus while it refreshes.

## Decision

`VectorStore` serializes writes and queries with one re-entrant lock. A source
replacement captures the old IDs, documents, embeddings, and metadata, writes the
new version, removes obsolete IDs, and publishes one revision. If any operation
fails, it deletes the attempted source version and restores the captured version.
The lock provides process-local all-or-nothing visibility; it is not a distributed
transaction across multiple API processes sharing a Chroma path.

`Bm25Index.refresh()` builds a complete immutable snapshot off to the side and
publishes it with a single locked reference swap. Queries retain the snapshot they
started with, so a refresh cannot mutate their corpus arrays.

## Consequences

- Ingest, rebuild, delete, and vector queries are serialized within one service process.
- Failed replacements preserve the previous source, subject to the underlying
  Chroma storage remaining available for rollback.
- Multi-process writers require an external ownership/locking boundary and are not
  supported by this contract.
