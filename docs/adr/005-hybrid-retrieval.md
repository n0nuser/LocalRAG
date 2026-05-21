# ADR 005: Hybrid retrieval (vector + BM25)

## Context

Pure vector search performs poorly on exact tokens such as error codes, SKUs,
and version identifiers. LocalRAG previously used only embedding similarity,
which made exact-string queries unreliable.

## Decision

Introduce a BM25 index over the same chunk corpus and run it in parallel with
vector retrieval:

- Build an in-memory BM25 index from chunks stored in Chroma.
- Refresh the BM25 index after successful ingestion updates.
- Fuse vector and BM25 rankings in the retriever using reciprocal rank fusion
  (RRF), with optional weighted tuning via settings.

## Consequences

- Exact-string retrieval quality improves without replacing the vector store.
- Query-time complexity increases slightly due to dual ranking paths.
- Retrieval remains configurable: users can keep vector-only mode if desired.
