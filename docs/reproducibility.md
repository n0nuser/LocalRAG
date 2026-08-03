# Reproducible evaluation runs

Benchmark numbers are only useful if you can tell a real regression from noise.
This page describes what LocalRAG pins, what it records, and what it cannot
control.

## Running a reproducible eval

```bash
# Full dataset, default seed (42)
uv run localrag eval --offline

# A deterministic 10-example subset, explicit seed
uv run localrag eval --offline --seed 7 --sample 10

# Same thing via the runner directly
uv run python evals/run_evals.py --offline --seed 7 --sample 10
```

Two runs with the same seed, the same dataset, and the same environment
evaluate **the same examples in the same order** and send the judge the same
sampling parameters.

## What the seed controls

- **Down-sampling** — `--sample N` draws a deterministic subset (via a seeded
  RNG) from the selected split's records, ordered by stable `record_id`. See
  [docs/eval-datasets.md](eval-datasets.md) for the dataset/split contract.
- **Judge sampling** — the RAGAS judge runs at `temperature=0.0` with the run's
  seed, so its verdicts are repeatable rather than re-rolled per run.
- `--seed` must be a non-negative integer; an invalid value fails the CLI
  argument parse rather than silently coercing.

## Making generation deterministic

The answering model is *not* pinned by default — Ollama's per-model defaults
apply and nothing is sent in the request's `options` block. To pin it:

```bash
LLM_TEMPERATURE=0.0
LLM_SEED=42
```

Both settings are optional. Leave them unset for normal use, where a little
sampling variety is usually desirable; set them when you need run-to-run
comparability. When either is set, LocalRAG sends an `options` block on every
`/api/chat` call; when both are unset, the block is omitted entirely.

## What each result file records

Every file in `evals/results/` carries a `dataset` block (identity) and an
`environment` block (provenance).

`dataset`:

| Field | Why it matters |
| --- | --- |
| `dataset_id`, `dataset_version`, `split` | Which dataset produced these scores |
| `checksum` | Content hash of the manifest — detects a silent edit even if the version tag wasn't bumped |
| `selected_record_ids` | The exact records evaluated, in order — the ground truth for "did these two runs evaluate the same inputs" |

`environment`:

| Field | Why it matters |
| --- | --- |
| `seed` | Reproduces down-sampling and judge sampling |
| `git_sha`, `git_dirty` | Ties results to code; `git_dirty: true` means they are **not** tied to a clean commit |
| `uv_lock_sha256` | Detects dependency drift between runs |
| `judge_model_digest`, `embedding_model_digest` | Model **tags are mutable** — `gemma3:4b` can be re-pulled and point at different weights. The digest is the real pin. |
| `python_version`, `platform_summary`, `cpu_count`, `total_ram_gb` | Hardware/OS differences that affect throughput and, on GPU, numerics |
| `settings_snapshot` | Retrieval and chunking knobs that change scores |

`settings_snapshot` is an explicit allowlist (see `SNAPSHOT_SETTINGS_FIELDS` in
`evals/environment.py`), not a full `Settings` dump — that keeps host paths and
credentials out of committed result files. **Add new retrieval-affecting
settings to that list** or runs will silently differ with no record of why.

## Known limits — what is *not* reproducible

Identical seeds do not guarantee identical scores:

- **GPU nondeterminism.** Floating-point reduction order on GPU is not stable
  across runs, and can differ across driver or hardware. CPU-only runs are more
  stable but slower.
- **Ollama seeding is best-effort.** `seed` reduces variance but is not a
  contractual guarantee of identical output across versions, quantizations, or
  context-window changes.
- **Model re-pulls.** Same tag, different weights. Compare digests, not names.
- **Live API mode.** Without `--offline`, results depend on the current index
  contents. Re-ingested or re-chunked corpora shift retrieval.
- **Concurrency.** Parallel scoring can interleave differently; ordering effects
  in batched embedding calls are not fully pinned.

Treat small score deltas (roughly <0.02) as noise unless they persist across
several seeds. When a regression looks real, check `git_sha`, the model
digests, and `settings_snapshot` before assuming the retrieval change caused it.
