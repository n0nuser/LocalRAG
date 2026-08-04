# ADR 032: Versioned Retriever Plugin Boundary

- **Status:** Accepted
- **Date:** 2026-08-04

## Context

LocalRAG needs extensions without freezing unstable contracts for retrievers,
rerankers, chunkers, and evaluators at once. The existing synchronous retriever
request and context shape is the smallest useful stable seam.

## Decision

The first plugin family is retrievers. `localrag.plugins.retriever` publishes a
`1.0` synchronous protocol: settings are injected into a selected factory and
`retrieve(question, n_results, metadata_filter)` returns retrieval contexts.
Plugins own resources and expose `close()`; the API closes the selected plugin
once during shutdown.

Discovery uses only Python package entry points in the `localrag.retrievers`
group. The built-in retriever is registered as `builtin` through the same
descriptor contract, preserving its existing runtime dependencies and behavior.
`retriever_plugin` selects exactly one ID. Discovery sorts IDs and rejects
duplicates, unknown IDs, malformed metadata, import failures, and unsupported
versions.

Contract versions are `MAJOR.MINOR`: major mismatches are rejected and minor
compatibility must be explicitly declared by the plugin. Plugin code is trusted
local code executed in-process. There is no sandbox or network/package
installation behavior; deployments must explicitly install and pin plugin
distributions. Optional plugin dependencies are declared by those distributions
and missing dependencies fail only when that plugin is selected.

## Non-goals

This ADR does not define plugins for rerankers, chunkers, evaluators, embeddings,
or datasets. It does not redesign RAGAS/manual-only evaluation, the dataset
registry, embedding providers, or result schemas. Async retrievers, arbitrary
module-path loading, URLs, marketplaces, and plugin isolation processes are
deferred.

## Consequences

The public contract can evolve independently of concrete built-ins, and plugin
selection is deterministic and isolated. Third-party code remains part of the
application trust boundary, and lifecycle failures can affect the process; pin
and review installed packages accordingly.
