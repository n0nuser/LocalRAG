# ADR 001: Vector Store — ChromaDB

**Status:** Accepted  
**Date:** 2026-04-30

## Context

LocalRAG needs a vector store to persist document embeddings and support semantic search. The store must work fully offline, require no external service accounts, and be simple enough to run on a developer laptop.

Candidates evaluated:

| Option | Notes |
| --- | --- |
| **ChromaDB** | Embedded Python library with persistent local storage; no server required |
| FAISS | Fast but purely in-memory — requires custom serialization for persistence |
| Weaviate | Capable but requires a running Docker service and is overkill for local use |
| Qdrant | Similar to Weaviate; strong cloud offering but heavier local footprint |
| pgvector | Requires PostgreSQL; adds a significant dependency for a local-first tool |

## Decision

Use **ChromaDB** (`chromadb` package) with `PersistentClient` storing data at `CHROMA_PERSIST_PATH`.

## Rationale

- **Zero-server setup**: `PersistentClient` writes directly to local disk — no daemon needed.
- **Python-native**: integrates as a library import, consistent with the local-first ethos.
- **Collection isolation**: named collections make it easy to support multiple knowledge bases.
- **Metadata filtering**: first-class support for metadata predicates used by `delete_by_source` and `list_distinct_sources`.

## Consequences

- **Positive**: simple deployment, no external dependency, fast iteration.
- **Negative**: not suitable for multi-node or high-concurrency workloads. If LocalRAG ever needs horizontal scaling, migrating to a separate Chroma server or Qdrant would be required.
- **Mitigation**: the `VectorStore` class in `localrag/storage/vector_store.py` is a thin adapter — swapping the backend requires changing only that file.
