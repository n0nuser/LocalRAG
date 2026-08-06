# ADR 040: Request-Scoped Collection Selection

## Status

Accepted

## Context

The HTTP query API previously always used the collection configured at process
startup. That made `GET /collections` expose namespaces that HTTP clients could
not query, while the CLI could select them with a settings override.

## Decision

`POST /query`, `/query/contexts`, and `/query/stream` accept an optional
`collection` field. An omitted field retains the configured
`chroma_collection_name`. A supplied name selects an existing Chroma collection
for that request by creating a request-scoped vector store, BM25 snapshot, and
retriever/engine view. Process-wide settings and the cached default engine are
not mutated.

The endpoint remains behind `API_KEY`. Collection names are not treated as
tenant authorization: deployments that use collections for tenant separation
must enforce per-key authorization at a gateway or provide separate persist
paths. This is an explicit single-deployment namespace selection feature, not a
multi-tenant isolation boundary.

## Consequences

- JSON, contexts, and SSE query paths can address any existing collection.
- A request-scoped hybrid query rebuilds the BM25 snapshot for the selected collection.
- Unknown collections return a query `404` rather than exposing a backend error.
- Retriever plugins that do not expose the built-in collection seam cannot use
  request-scoped selection.
