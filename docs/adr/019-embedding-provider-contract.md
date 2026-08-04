# ADR 019: Embedding Provider Contract and Collection Compatibility

**Status:** Accepted  
**Date:** 2026-08-04

## Context

Embedding vectors are persisted in Chroma and are used by both ingestion and
retrieval. A provider or model change changes the vector space, so treating
embedding as an Ollama HTTP helper can silently corrupt retrieval.

## Decision

`localrag.embedding.EmbeddingProvider` is the single contract. Providers expose
`embed` and `embed_batch`, return finite non-empty vectors in input order, and
return `[]` for an empty batch without network work. Providers own batching;
`batch_size` is an optional positive hint. They expose provider name, model,
dimension (when known), timeout, and `close()` lifecycle methods. Typed errors
distinguish configuration, transport, malformed responses, and collection
incompatibility.

The factory selects `EMBEDDING_PROVIDER`, defaulting to `ollama`. The effective
model is `EMBEDDING_MODEL` when set, otherwise the legacy `OLLAMA_EMBED_MODEL`,
so existing `.env` files retain their behavior. `SentenceTransformersProvider`
is supported through the optional `embedding` extra and is never imported by
the default Ollama path.

Every factory-created provider instance is shared by ingestion and retrieval
through API dependency injection. Collection metadata records provider, model,
and dimension. Runtime identity and dimension must match before query or
upsert; a mismatch fails with an actionable rebuild error. Unannotated legacy
collections are adopted only after the first validated embedding operation.
Changing the embedding space requires the explicit collection rebuild command;
vectors are never mixed silently.

## Consequences

Provider-specific types stay behind the factory and provider implementations.
Embedding caching and benchmark profiles remain separate future seams. Optional
providers must remain optional so the default install stays local Ollama-only.
