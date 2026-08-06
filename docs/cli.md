# CLI

The `localrag` CLI accesses the configured local Chroma filesystem directly. It
does not use the HTTP API and therefore does not send `X-API-Key`; protect the
filesystem and `.env` with normal operating-system permissions.

## Configuration

Use the global options before a command:

```bash
uv run localrag --config config.yaml query "What is indexed?"
uv run localrag --config config.yaml --set rag_top_k=10 config-show
```

Resolution order is built-in defaults, YAML, `.env`, process environment, then
explicit `--set FIELD=VALUE` overrides. YAML is strict and supports the sections
shown in [`config.example.yaml`](../config.example.yaml). `${ENV_NAME}` values
are interpolated from the environment. Relative YAML paths are resolved against
the YAML file directory. API startup uses the same loader through
`LOCALRAG_CONFIG=./config.yaml`; invalid or missing selected files fail before
services are constructed.

`config-show` prints a deterministic resolved snapshot. API keys are redacted,
path values are represented as `<path>`, and document contents or the full
environment are never included. Keep credentials in environment variables or
an untracked secrets mechanism, not YAML.

## Ingest

`ingest` takes a file or a directory and reports each file as it completes:

```bash
uv run localrag ingest ./docs
[1/3] guide.md — 42 chunks
[2/3] notes.md — skipped (no chunks)
[3/3] report.pdf — 117 chunks
status=ok files_processed=2 total_chunks=159
```

Progress lines go to **stderr** and the final `status=` summary to **stdout**, so
piping stdout stays parseable. Pass `--quiet` to suppress the per-file lines and
print only the summary.

Files that no parser can handle — and binary content in a file with a textual
extension — are reported and skipped rather than being read as text. See
[document-formats.md](document-formats.md).

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
