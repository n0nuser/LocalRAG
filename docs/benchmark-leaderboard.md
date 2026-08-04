# Benchmark Leaderboard

The leaderboard is a publication layer, not a benchmark runner. It consumes
reviewed canonical result artifacts produced by the matrix/evaluation flows
(#73 and #84) and writes a deterministic Markdown table. It never calls a
model, samples a dataset, or fills missing measurements.

## Contract

Each input is a JSON object with `schema_version: 1`, `source_kind`
(`matrix_case` or `evaluation_result`), and a stable `artifact_id`. It must
include dataset ID/version/split/hash; evaluation schema; exact model provider,
name, revision, digest, quantization, and runtime; exact embedding name,
revision, and digest; retrieval, reranker, and chunker configurations; OS,
CPU, GPU, and RAM; cold/warm state; at least three repetitions and a warm-up
count; metric definitions and mean/median/p95/stddev with units; latency and
resource measurement methods; and timestamp, code revision, dependency lock
hash, and seed.

The machine-readable output is `schema_version: 1` with validated rows in
deterministic identity order. Markdown contains the same measured values and
provenance. It contains no generation timestamp, so unchanged inputs produce
byte-identical output.

## Generation

```bash
uv run localrag leaderboard \
  evals/results/publications/model-a.json \
  evals/results/publications/model-b.json \
  --output leaderboard.md \
  --json-output leaderboard.json
```

Empty input produces an informative empty table. Any supplied malformed,
incomplete, duplicate, non-finite, or incompatible artifact fails before
publication. Rows must share dataset identity, evaluation schema, embedding,
retrieval/reranker/chunker configuration, temperature, metric definitions, and
measurement units. Cold and warm rows are never combined.

An optional exact model matrix checks that every requested identity is present.
It uses immutable identities, not vague model families:

```json
{"schema_version": 1, "identities": [
  {"provider": "ollama", "name": "gemma3:4b", "revision": "rev-1", "digest": "sha256:..."}
]}
```

Review generated diffs and source artifacts together. Do not check in generated
benchmark output unless explicitly reviewed as a fixture. Regenerate only via
an intentional command or reviewed CI publication job. The run timestamp and
code revision make stale entries visible; an old table is preferable to an
unreviewed claim.

## Methodology And Limits

Repetitions exclude the documented warm-up count. Warm measurements exclude
one-time model/index/cache setup; cold measurements include it. p95 and
standard deviation are descriptive summaries, not confidence intervals.
Quality definitions and direction remain part of the source contract even when
the table does not rank rows automatically.

Compare only rows passing compatibility checks. Model tags are not identity:
compare immutable digests and revisions. Different hardware, runtimes,
dependency locks, seeds, dataset hashes, retrieval settings, or temperature
states are not interchangeable. GPU and provider nondeterminism can still
produce variation. No results are published by this change, and no model result
should be inferred from an empty table.

The publication schema is stricter than historical #84 result files where
these fields were not recorded. Producers must add the missing provenance; the
leaderboard never guesses it.
