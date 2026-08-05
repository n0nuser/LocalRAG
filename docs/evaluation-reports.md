# Evaluation Reports

> Part of the [evaluation documentation](evaluation.md) — start there for the pipeline overview.

Generate a local report from one or more canonical benchmark artifacts:

```bash
uv run localrag report evals/results/matrices/fixture/manifest.json -o report.html
uv run localrag report run-a.json run-b.json --strict
```

The consumer accepts the `MatrixManifest` contract from `evals/matrix.py` and the
versioned `ResultFile` contract from `evals/results/schema.py`. It does not define
or migrate a competing evaluation schema. `report.html` is the only output file
and is overwritten on each invocation. Its embedded JSON and JavaScript/CSS are
deterministic for the same inputs.

See [ADR 013](adr/013-versioned-benchmark-results.md) and [ADR 015](adr/015-canonical-benchmark-matrix.md).

The page shows run and dataset identity, configuration, metric scores and
thresholds, available/unavailable status, per-case failures, and latency/resource
fields when present. Missing and non-finite values remain unavailable. Different
dataset identities are rendered but marked incompatible; the report does not
pretend those scores are comparable. Malformed files are listed while valid files
continue to render. Empty input produces a valid empty page. `--strict` changes
input errors into exit code 1 after the report is written.

Canonical evaluation results may also contain failure analysis. The report shows
stable case IDs, primary/secondary labels, confidence, and deduplicated counts;
it does not show raw answer, question, context, document, or source-path content.

No CDN, remote font, image, JavaScript, or stylesheet is used. The file can be
opened offline. Questions, answers, contexts, and source paths are omitted by
default for privacy. Input metadata and failure messages are treated as untrusted
and escaped. Reports should still be kept private because model/configuration
metadata may be sensitive. This feature is an inspection view only; it is not
experiment tracking (#75) or leaderboard publication (#65).

Interpret scores only across runs with matching dataset ID, version, split,
checksum, metric contract, and relevant configuration. A threshold is a declared
interpretation aid, not a replacement for checking missing/error counts.
