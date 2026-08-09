# Evaluation dataset contract

> Part of the [evaluation documentation](evaluation.md) — start there for the pipeline overview.

The durable decision behind this contract is [ADR 011](adr/011-evaluation-dataset-contract.md).

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
- **Retrieved IDs**: `offline_retrieved_citation_ids` if set, else every
  declared `citation_id`.

The default for retrieved IDs asserts that retrieval returned everything the
record declares — perfect recall. That is the right default for a record whose
citations *are* its context, but it is an assertion, not a measurement, and it
makes retrieval failure inexpressible. A record whose point is that retrieval
missed something must list what was actually retrieved in
`offline_retrieved_citation_ids`; see the `localrag-scope` fixture.

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
| `localrag-scope` | binary | 4 | `default` |

`localrag-core` is the project's main fixture, migrated from the legacy flat
`evals/dataset.json` (now removed). `localrag-graded` is a minimal
second dataset that exists to prove the registry supports more than one
dataset and judgment type without runner changes.

`localrag-scope` is the regression fixture for
[#174](https://github.com/n0nuser/LocalRAG/issues/174): questions whose
temporal qualifier — a single occurrence versus repeated exposure — decides
which passage answers them. Embedding similarity keys on topic, so a question
about one occurrence retrieves the passages about cumulative effects; both are
topically "effects of X".

**It is expected to fail, and that is the point.** `retrieval_recall` scores
**0.375** against a threshold of 0.8. The fixture exists to make the failure a
number a later retrieval-tuning change can be judged against, and to catch a
fix that helps this shape of query while hurting others.

Its corpus is synthetic and authored for the fixture, describing a fictional
device. The motivating evidence came from a copyrighted book; the bundled
fixtures are CC0, and shipping excerpts to reproduce a ranking bug is not worth
the licensing question when an authored corpus reproduces the same shape.

| Record | Shape | `retrieval_recall` |
| --- | --- | --- |
| `single-overvoltage-event-effect` | Acute question, only chronic passages retrieved | 0.0 |
| `repeated-overvoltage-events-effect` | Chronic question, chronic passages retrieved | 1.0 |
| `clamp-cooldown-interval` | Front matter and legal boilerplate outrank the answer | 0.0 |
| `single-event-recovery-procedure` | One of two relevant passages retrieved | 0.5 |

The control record matters as much as the failing ones: without it a metric
that returned zero unconditionally would look like a successful reproduction.
The partial record proves the metric is graded rather than binary.

## Metric annotations

Citation IDs are stable within a record and are the only valid join key for
citation metrics. Relevance judgments identify the relevant IDs; an evaluator
must also provide citation IDs for the answer being scored. If either side is
absent, citation accuracy is explicitly unavailable rather than an invented
score. Reference answers may be supplied in `reference_answers` for
multi-reference EM/F1; the legacy `reference_answer` remains the fallback.

Metric formulas, thresholds, and missing-data behavior are documented in
[evaluation-metrics.md](evaluation-metrics.md).

## What this contract does not cover

- **Result comparison / regression gating across runs** — see #84.
- **Per-case failure artifacts** — see #86.

Result and metric consumers are specified by [ADR 013](adr/013-versioned-benchmark-results.md) and [ADR 014](adr/014-evaluation-metric-contract.md).
