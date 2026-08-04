# Reproducible evaluation runs

Benchmark numbers are only useful if you can tell a real regression from
noise. This page defines what LocalRAG guarantees about a run, the full field
reference for what gets recorded, and what it explicitly does not promise.

See [ADR 012](adr/012-reproducible-evaluation-metadata.md) for the architectural decision.

## Embedding cache benchmark

The cache has a separate manual benchmark because it measures provider calls and
storage behavior, not answer quality:

```bash
uv run python benchmarks/embedding_cache_benchmark.py corpus/*.md \
  --cache-path ./data/embedding-cache-benchmark --model nomic-embed-text
```

It clears the selected cache, runs the same sorted corpus once cold and once warm,
and emits JSON containing provider/model/revision, Python/platform/CPU metadata,
provider calls, hit/miss counts, wall time, p50/p95 per-input latency, and cache
bytes. Measurements are descriptive; the workflow makes no universal speedup
claim. The cache is ingestion-only and does not alter the manual RAGAS workflow.

## Reproducibility levels

Two different things get called "reproducible" and conflating them leads to
false confidence:

1. **Input/config reproducibility** (guaranteed). Given the same dataset
   identity, split, seed, and settings, two runs select the **same records in
   the same order** and construct the **same judge call parameters**
   (temperature, seed). This is fully deterministic and tested.
2. **Model output determinism** (best-effort, not guaranteed). Whether the
   *model's actual output text* is identical across runs depends on hardware,
   drivers, kernel versions, quantization, and the provider's own seeding
   implementation. LocalRAG pins what it can (see below) and records enough
   metadata to *detect* when something in that chain moved — it does not
   promise bit-for-bit identical answers.

Every guarantee below is level 1 unless stated otherwise.

## Running a reproducible eval

```bash
# Full split, default seed (42)
uv run localrag eval --offline

# A deterministic 10-example subset, explicit seed, specific dataset/split
uv run localrag eval --offline --seed 7 --sample 10 --dataset localrag-core --split smoke

# Same thing via the runner directly
uv run python evals/run_evals.py --offline --seed 7 --sample 10
```

See [docs/eval-datasets.md](eval-datasets.md) for `--dataset`/`--version`/`--split`.

## Canonical benchmark matrices

Use the manually invoked matrix command when comparing configurations:

```bash
uv run localrag benchmark --profile fixture --dry-run
uv run localrag benchmark --profile embedding-comparison
uv run localrag benchmark --profile hyde --dataset localrag-core --seed 42
uv run localrag benchmark --matrix path/to/matrix.json
```

`evals/matrix.py` is the versioned JSON contract. Dimension values are validated
against the capabilities currently exposed by LocalRAG (`ollama`,
`nomic-embed-text`, `gemma3:4b`, `hybrid`/`vector`, and the installed chunking
options). Expansion sorts dimension names and values before computing Cartesian
products, so reordered input produces the same case IDs and order. Each case
gets its own work directory and a structured result with latency and resource
units, metrics or an error, and artifact paths. The matrix manifest records run
and matrix IDs, dataset and corpus checksums, revision/dirty state, model and
provider identity, effective configuration, supported dimensions, seed, and
timestamps.

The matrix decision is recorded in [ADR 015](adr/015-canonical-benchmark-matrix.md), and bounded execution in [ADR 016](adr/016-bounded-parallel-evaluation.md).

The `hyde` profile is a four-arm smoke comparison (`baseline`, `rewrite`, `hyde`,
`rewrite+hyde`) on the same #73 dataset split and seed. Keep retrieval quality,
generation latency, and retrieval latency as separate reported fields; it is an
experiment scaffold, not evidence of a universal improvement.

Dry runs exit `0`. A configuration or validation error exits `2` before any case
starts. Execution continues across independent cases; if one or more cases fail,
the manifest is still written and the command exits `1`. The existing
`localrag eval` single-run path remains unchanged. No evaluation workflow is
triggered automatically.

## Seed precedence

| Priority | Source | Example |
| --- | --- | --- |
| 1 (highest) | `--seed` CLI flag | `--seed 7` |
| 2 | `EVAL_SEED` environment variable | `EVAL_SEED=7` |
| 3 (default) | Built-in default | `42` |

A non-integer `EVAL_SEED` is a configuration error and raises rather than
being silently ignored. A negative resolved seed (from any source) fails the
CLI argument parse before any work starts.

### What the seed controls — per-operation coverage

Not every random-ish operation in the eval path is actually seed-controlled.
Each result file's `environment.seed_coverage` states this explicitly per
operation, rather than leaving it implicit:

| Operation | Seed-controlled | Notes |
| --- | --- | --- |
| `record_downsampling` | Yes | `--sample N` draws from the split via `random.Random(seed)` |
| `judge_llm_sampling` | Yes | RAGAS judge runs at `temperature=0.0` with the run's seed |
| `answering_model_sampling` | No | Controlled separately by `LLM_TEMPERATURE`/`LLM_SEED` (see below), not `--seed` |
| `embedding_computation` | No | No seed concept applies — embeddings are a deterministic function of fixed input text |

## Making generation deterministic

The answering model is *not* pinned by default — Ollama's per-model defaults
apply and nothing is sent in the request's `options` block. To pin it:

```bash
LLM_TEMPERATURE=0.0
LLM_SEED=42
```

Both settings are optional and independent of `--seed`/`EVAL_SEED` above —
they affect the model *answering* a query, not the eval runner's own
selection/judge behavior. Leave them unset for normal use, where a little
sampling variety is usually desirable; set them when you need run-to-run
comparability. When either is set, LocalRAG sends an `options` block on every
`/api/chat` call; when both are unset, the block is omitted entirely.

## Field reference

Every file in `evals/results/` is a versioned result document with
`schema_version`, `run_id`, `timestamp`, `dataset`, `selected_ids`, `metrics`,
`provenance`, and `status`. The canonical schema and migration code live in
`evals/results/schema.py`; the pre-#84 shape (top-level `scores` and
`environment`) is explicitly migrated as version 0. Unknown future versions
fail rather than being guessed at.

### `dataset` (identity — see [docs/eval-datasets.md](eval-datasets.md))

| Field | Type | Meaning |
| --- | --- | --- |
| `dataset_id`, `dataset_version`, `split` | string | Which dataset produced these scores |
| `checksum` | string | sha256 over the manifest content (excludes the `schema_version` bookkeeping field) — detects a silent edit even if the version tag wasn't bumped |
| `selected_ids` | list[string] | The exact records evaluated, in order |

### Metrics and comparison

Each metric has a descriptor with `direction` (`higher_is_better` or
`lower_is_better`), an optional `threshold`, optional `unit`, and a
`missing_value` policy. Non-finite values are represented as missing and are
listed in `non_finite_cases`; they are never changed to zero. Per-case values
are retained in `cases` when the evaluator provides them.

Compare only explicitly selected baselines:

```bash
uv run python evals/compare.py evals/results/run.json --baseline evals/baselines/default.json
uv run localrag eval-compare evals/results/run.json --baseline-name default --threshold faithfulness>=0.60
```

Thresholds are `metric>=number`, `metric<=number`, or
`metric_delta>=number`/`metric_delta<=number`. The metric must exist; malformed
expressions and unknown metrics are usage errors. Comparison reports added or
removed metrics, missing or extra case IDs, non-finite values, absolute and
relative deltas, and incompatible dataset/provenance inputs. Exit codes are
0 for a passing comparable result, 1 for a comparable regression or failed
threshold, and 2 for usage, missing/schema, or incompatible inputs.

Baselines are reviewed JSON artifacts in `evals/baselines/`. Update one by
running a deliberate evaluation, reviewing its dataset checksum, selected
IDs, metric descriptors, and provenance, then replacing the named artifact in
a separate change. The manual `Evaluation comparison` workflow invokes this
command against the explicitly supplied baseline; RAGAS remains a manually
dispatched evaluation step, not an automatic CI dependency.

### `environment` (provenance)

Most fields are wrapped in a **capability envelope**:

```json
{"value": ..., "status": "available" | "unsupported" | "unavailable", "reason": "..." | null}
```

- **`available`** — `value` is populated; `reason` is `null`.
- **`unsupported`** — the field doesn't apply to this run at all (e.g. no GPU
  present). Not a failure.
- **`unavailable`** — the field *should* exist but couldn't be read (e.g.
  Ollama unreachable, git not installed, model not pulled). A real gap —
  `value` is always `null` here, never a fabricated guess.

This distinguishes three states a bare `null` conflates: "not applicable
here," "should be there but isn't right now," and "we didn't even check."

| Field | Wrapped? | Meaning |
| --- | --- | --- |
| `metadata_schema_version` | no | Version of this metadata shape. Bumped only on incompatible (non-additive) changes; a reader keyed on a version can trust every field that existed at that version. |
| `seed`, `seed_source` | no | Resolved seed and where it came from (`cli`/`config`/`default`) |
| `seed_coverage` | no | Per-operation seed coverage, see above |
| `git_sha` | yes | Commit the run was produced from |
| `git_dirty` | yes | `true` means uncommitted changes were present — **results are not tied to a clean commit** |
| `uv_lock_sha256` | yes | Hash of `uv.lock` — detects full dependency-graph drift |
| `package_versions` | no (dict) | Installed versions of packages that can change judge/embedding/generation behavior (see `TRACKED_PACKAGES` in `evals/environment.py`) |
| `python_version`, `platform_summary` | no | Interpreter and OS/arch string |
| `cpu_count`, `total_ram_gb` | yes | Host sizing |
| `gpu` | yes | `nvidia-smi` name/driver/memory per GPU, when present. `unsupported` (not `unavailable`) on hosts with no NVIDIA GPU — that's the common case, not a gap. |
| `judge_model`, `embedding_model` | no | Model tags requested |
| `judge_model_digest`, `embedding_model_digest` | yes | **Tags are mutable** — `gemma3:4b` can be re-pulled and point at different weights. The digest is the real pin. Required-when-available: Ollama can always expose a digest for a pulled model, so a miss here means Ollama was unreachable or the model isn't pulled — always `unavailable`, never `unsupported`. |
| `settings_snapshot` | no (dict) | Allowlisted retrieval/chunking/generation settings, see below |

### Redaction policy

`settings_snapshot` is an **explicit allowlist**
(`SNAPSHOT_SETTINGS_FIELDS` in `evals/environment.py`), not a full `Settings`
dump. Secret-bearing fields (`api_key`, `anthropic_api_key`,
`openai_api_key`) and host-specific paths (`chroma_persist_path`,
`upload_dir`, `audit_log_path`) are never in the allowlist and are tested to
stay out. **Add new retrieval-affecting settings to the allowlist** or runs
will silently differ with no record of why — but never add a
credential or absolute-path field to it.

Nothing in the metadata payload includes record question/answer content
beyond what's already in `dataset.selected_record_ids` (IDs only, not text).

## Known limits — what is *not* reproducible

Identical seeds and identical `environment` blocks do not guarantee
identical scores:

- **GPU nondeterminism.** Floating-point reduction order on GPU is not stable
  across runs, and can differ across driver or hardware. CPU-only runs are
  more stable but slower. The `gpu` field records what was present; it does
  not neutralize this.
- **Ollama seeding is best-effort.** `seed` reduces variance but is not a
  contractual guarantee of identical output across Ollama versions,
  quantizations, or context-window changes.
- **Model re-pulls.** Same tag, different weights. Compare digests, not
  names — that's exactly why digests are recorded.
- **Live API mode.** Without `--offline`, results depend on the current index
  contents. Re-ingested or re-chunked corpora shift retrieval.
- **Concurrency.** Parallel scoring can interleave differently; ordering
  effects in batched embedding calls are not fully pinned.

Treat small score deltas (roughly <0.02) as noise unless they persist across
several seeds. When a regression looks real, check `git_sha`, the model
digests, `package_versions`, and `settings_snapshot` before assuming a
retrieval change caused it — comparing two results with a mismatched
`dataset.checksum` is comparing different inputs, not a regression.

Result comparison and CI regression gating are implemented by
`evals/compare.py` and documented above.

The canonical result and migration decision is [ADR 013](adr/013-versioned-benchmark-results.md).

## Optional experiment tracking

MLflow is an optional best-effort mirror, not a replacement for local JSON:

```bash
uv sync --extra tracking
EVAL_TRACKING_ENABLED=true uv run localrag benchmark --profile fixture
```

The default URI is the local `file:./evals/tracking` store; set
`EVAL_TRACKING_URI` for a local MLflow server. Matrix runs create one stable
parent and nested stable case runs. Failed cases and backend failures remain
visible in the canonical manifest, while tracking failures are isolated. Only
canonical JSON artifacts are selected deterministically. Secrets, paths,
prompts, documents, contexts, questions, and answers are redacted unless the
explicitly documented `EVAL_TRACKING_CAPTURE_CONTENT=true` opt-in is enabled.
See [ADR 018](adr/018-optional-mlflow-experiment-tracking.md).

Leaderboard publication is a separate, strict consumer of reviewed canonical
artifacts. See [benchmark-leaderboard.md](benchmark-leaderboard.md): it rejects
results that lack required provenance and cold/warm repetition fields and never
runs a benchmark while generating a table.
