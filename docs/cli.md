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

The summary reports the run's outcome, and the exit code follows it:

| `status=` | Meaning | Exit code |
| --- | --- | --- |
| `ok` | Every discovered file was processed or deliberately skipped | 0 |
| `partial` | Some files ingested, at least one failed permanently | 1 |
| `error` | Nothing was ingested | 1 |

When any file fails, the summary gains `failed=<n>` and each failure is listed on
stderr. A run that ingested nothing never reports success, so scripts and CI can
rely on the exit code.

Files that no parser can handle — and binary content in a file with a textual
extension — are reported and skipped rather than being read as text. See
[document-formats.md](document-formats.md).

`ingest` accepts a file or a directory and writes into the configured Chroma
persist path:

```bash
uv run localrag ingest ./docs
uv run localrag ingest ./docs/guide.md
```

**One writer per persist path.** Chroma's embedded client keeps its HNSW
segments in per-process memory with no cross-process invalidation, so two
processes writing the same `CHROMA_PERSIST_PATH` corrupt each other's view —
the boundary [ADR 035](adr/035-atomic-ingestion-replacement.md) already declares
out of contract. Every ingest therefore takes an exclusive `flock` on
`<CHROMA_PERSIST_PATH>/.ingest.lock` for the duration of the run, and the same
lock covers collection rebuilds.

A second concurrent ingest **fails immediately** rather than queueing — an
ingest can run for minutes, so a caller is better told to retry than left
hanging:

```
status=error reason=concurrent_ingest detail=another ingest is already running against ./data/chroma; wait for it to finish or use a different collection
```

The message goes to stderr and the command exits `1`. The equivalent HTTP
ingest endpoints return `409 Conflict` with the same detail. Query and other
read paths are never locked. The lock is advisory: if the persist directory
cannot be opened, or the filesystem does not support `flock` (some network
mounts) the ingest proceeds unguarded and logs a warning.

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
