# ADR 011: Versioned evaluation dataset contract

## Context

Evaluation metrics and benchmark comparisons are meaningless if the inputs can
change without being identifiable. A flat fixture also cannot express stable
record selection, citation annotations, or offline substitutes without making
those conventions implicit in the runner.

## Decision

Represent evaluation inputs as strict, registered `DatasetManifest` documents.
`dataset_id` identifies the logical dataset and `dataset_version` identifies an
immutable snapshot. The manifest checksum is recorded with every result and
detects edits even when a version tag was not bumped.

Records have stable `record_id` values. Splits name their membership and
declared ordering by record ID; selection and result documents use IDs rather
than file positions. Records may carry references, record-scoped citation IDs,
binary or graded relevance judgments, and optional multi-reference answers.
Unknown fields, duplicate IDs, dangling references, invalid judgment types,
missing graded scores, invalid splits, and unsupported schema versions fail
before scoring.

The registry discovers JSON fixtures without runner changes. Offline mode uses
declared `offline_answer` and `offline_contexts`, falling back to the reference
answer and citation text. Missing required offline artifacts are an explicit
error, not an empty-context score.

Dataset contracts precede metrics and benchmarks: they establish identical
inputs and annotation joins before a score, regression, or matrix case can be
interpreted.

## Consequences

- Dataset evolution requires a new immutable version and preserves auditability.
- Sampling and comparisons can name exactly which records were evaluated.
- Offline evaluation is useful without silently fabricating context or answers.
- Dataset authors must maintain checksums, annotations, and split membership as
  part of the evaluation contract.

## Related

[Evaluation datasets](../eval-datasets.md), [ADR 013](013-versioned-benchmark-results.md), [ADR 014](014-evaluation-metric-contract.md)
