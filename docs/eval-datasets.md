# Evaluation dataset contract

The eval runner (`evals/run_evals.py`) does not read a fixed file. It selects
a dataset from a registry by ID, version, and split. This page covers the
manifest format, how to add a dataset, and what the registry guarantees.

## Concepts

| Term | Meaning |
| --- | --- |
| `dataset_id` | Stable identity for a logical dataset (e.g. `localrag-core`). Never changes across versions. |
| `dataset_version` | Immutable snapshot tag (e.g. `1.0.0`). Editing records without bumping this breaks any result file that recorded the old checksum. |
| `split` | A named, ordered subset of a version's records (e.g. `default`, `smoke`). |
| `record_id` | Stable ID for one Q/A example. Selection, ordering, and result files key off this — never off list position in the manifest file. |
| `citation` | A source passage a record's judgments point at, scoped to that record. |
| `judgment` | Ground-truth relevance of one citation, `binary` (relevant: true/false) or `graded` (plus a numeric `grade`). |

Schema: `evals/dataset/schema.py` (`DatasetManifest`, `DatasetRecord`,
`Citation`, `RelevanceJudgment`, `DatasetSplit`). All models use
`extra="forbid"` — an unrecognized field fails validation rather than being
silently dropped.

## Adding a dataset

1. Write a manifest JSON file matching `DatasetManifest`. See
   `evals/dataset/fixtures/localrag-core-1.0.0.json` for a full example, or
   `localrag-graded-1.0.0.json` for a `judgment_type: graded` one.
2. Drop it in `evals/dataset/fixtures/` — `discover_fixtures()` globs `*.json`
   in that directory at import time, so **no runner code changes** to make it
   selectable.
3. Select it: `uv run localrag eval --dataset <id> --version <version> --split <split>`.

A manifest failing validation raises `DatasetValidationError` with the
specific field/record that's wrong — duplicate `record_id`s, a judgment
pointing at an unknown `citation_id`, a split referencing an unknown
`record_id`, an unrecognized `judgment_type`, a `graded` judgment missing its
`grade`, or an unsupported `schema_version` all fail at load time, before any
scoring happens.

## Offline mode

`--offline` never calls the API. Per record:

- **Answer**: `offline_answer` if set, else `reference_answer`.
- **Contexts**: `offline_contexts` if set, else each citation's `text`.

A record with no citations and no `offline_contexts` override has nothing to
score context-based metrics against — `_build_rows` raises
`OfflineArtifactsMissingError` naming the record, rather than silently
scoring against an empty context list.

## Selection and reproducibility

Records in a split are evaluated in the split's declared `record_ids` order
(not the manifest file's record order). `--sample N` draws a deterministic
subset via a seeded RNG (`--seed`, default 42) — see
[docs/reproducibility.md](reproducibility.md). Two runs with the same
dataset, version, split, seed, and sample size always select the same
`record_id`s in the same order; that guarantee does not extend to live-API
answers or model output.

Each result file's `dataset` block records `dataset_id`, `dataset_version`,
`split`, a content `checksum` (sha256 over the manifest, excluding the
`schema_version` bookkeeping field), and the exact `selected_record_ids` —
enough to tell whether two runs actually evaluated the same inputs.

## Bundled fixtures

| `dataset_id` | `judgment_type` | Records | Splits |
| --- | --- | --- | --- |
| `localrag-core` | binary | 23 | `default` (all), `smoke` (first 3) |
| `localrag-graded` | graded | 2 | `default` |

`localrag-core` is the project's main fixture, migrated from the legacy flat
`evals/dataset.json` (now removed). `localrag-graded` is a minimal
second dataset that exists to prove the registry supports more than one
dataset and judgment type without runner changes.

## What this contract does not cover

- **Metric formulas and thresholds** — see #74 (advanced metrics) once landed.
- **Result comparison / regression gating across runs** — see #84.
- **Per-case failure artifacts** — see #86.
