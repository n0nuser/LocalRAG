# Retriever Plugins

LocalRAG currently supports plugins for **retrievers only**. A plugin is trusted
Python code executed in the LocalRAG process; this mechanism is not a sandbox.
Install plugin distributions explicitly and pin their versions. LocalRAG never
downloads packages or imports arbitrary configured paths.

## Contract

The public imports are `localrag.plugins.retriever.RetrieverPlugin`,
`RetrievalContext`, and `CONTRACT_VERSION`. The current contract version is
`1.0`. A plugin exports an object through the `localrag.retrievers` entry-point
group with `plugin_id`, `contract_version`,
`compatible_contract_versions`, and `create(settings)`. The factory receives
`Settings` and returns a synchronous object implementing:
`retrieve(question, n_results=None, metadata_filter=None) -> list[RetrievalContext]`.

Each context contains stable `text`, `source`, `chunk_index`, `score`, and
`metadata` fields; `distance`, `ingested_at`, and `freshness_factor` are
optional. `close()` is required for resource ownership and is called once when
the API shuts down. Plugin failures, including missing optional dependencies,
are reported as typed plugin errors.

## Installation And Selection

The only discovery mechanism is Python package entry points in the
`localrag.retrievers` group. The example distribution is in
`examples/retriever-plugin`. Install it in the same environment as LocalRAG,
then select it with:

```yaml
retrieval:
  plugin: example
```

The equivalent environment setting is `RETRIEVER_PLUGIN=example`. The default
is `builtin`, which preserves existing behavior. Discovery is deterministic by
plugin ID. Duplicate IDs, unknown selections, malformed metadata, incompatible
versions, and import/factory failures fail clearly. Only the selected factory
is constructed.

## Version And Dependency Policy

Contract versions use `MAJOR.MINOR`. A major mismatch is rejected. Minor
compatibility is accepted only when the plugin explicitly lists the running
contract in `compatible_contract_versions`. Plugin dependencies belong in the
plugin distribution and remain absent from the core install. RAGAS/manual-only
evaluation, embeddings, rerankers, chunkers, and other plugin families are
deliberately outside this first boundary.
