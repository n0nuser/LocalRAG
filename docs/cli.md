# CLI

The `localrag` CLI accesses the configured local Chroma filesystem directly. It
does not use the HTTP API and therefore does not send `X-API-Key`; protect the
filesystem and `.env` with normal operating-system permissions.

## Inspect

`inspect` is read-only and never creates a missing collection, calls an LLM, or
uses a network provider:

```bash
uv run localrag inspect --collection localrag --sample-count 5 --format table
uv run localrag inspect --collection localrag --sample-count 20 --max-chars 500 --format json
```

JSON has schema version `1` and deterministic keys: `collection`,
`vector_count`, `document_count`, `collection_metadata`, `metadata_summary`, and
`samples`. Samples are sorted by chunk ID. Values are sanitized and bounded by `--max-chars`; metadata
summaries scan at most 1,000 metadata records. Sensitive metadata keys are
redacted. An empty collection succeeds with zero counts and no samples. Missing
collections return exit code `2`; storage failures return `1`; invalid options
return `2`.

## Benchmark

`benchmark` is only an adapter to the canonical `evals.matrix.run_matrix`
runner. It does not expand cases, aggregate metrics, or generate reports:

```bash
uv run localrag benchmark --profile fixture --dataset localrag-core \
  --result-output evals/results/matrices --dry-run
uv run localrag benchmark --config matrix.json --result-output evals/results/matrices
```

`--matrix` is an alias for `--config`; `--seed` overrides the matrix seed and
offline mode is enabled by default. The runner owns partial-failure policy and
the canonical manifest/result schema. Its artifact paths are printed unchanged.
Invalid configuration returns `2`, a malformed runner handoff returns `1`, and
the runner's own exit status is returned for execution or partial-case failure.
