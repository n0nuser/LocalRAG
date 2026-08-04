# ADR 018: Optional MLflow experiment tracking

## Context

Evaluation already produces local, versioned JSON and the matrix runner defines
stable parent and case IDs. An experiment index is useful, but it must not make
the offline installation depend on a service, change evaluation results, or
turn private evaluation inputs into telemetry. MLflow is Apache-2.0 licensed,
has a Python client, and supports a local `file:` tracking URI as well as a
locally hosted server.

## Decision

Use MLflow as the first and only tracking backend. It is available only through
the `tracking` optional extra (`uv sync --locked --extra tracking`). The default install
does not import or require MLflow. `EVAL_TRACKING_ENABLED=false` is the default;
when enabled, `EVAL_TRACKING_URI` may point at a local file store or server.

The internal `TrackingSession` boundary owns starting a parent run, starting
and ending child case runs, logging allowlisted metrics and artifacts, recording
status/errors, and cleanup. Provider-specific types do not cross this boundary.
The matrix `run_id` is the parent ID and each stable #73 `case_id` is its nested
child ID. Canonical `manifest.json` and `result.json` files are selected in
deterministic name/path order; arbitrary files are not uploaded by default.

Local JSON, validated by the #84 `ResultFile` contract where applicable, remains
the authoritative record. Tracking is a best-effort mirror: a missing optional
dependency, backend outage, retry exhaustion, or cleanup error is logged and
disables tracking without changing local files, case continuation, or exit
codes. Failed cases remain failed in both local JSON and the child status.

## Privacy boundary

Params are an explicit safe projection. API keys, tokens, credentials,
credential-bearing URLs, filesystem paths, prompts, questions, answers,
contexts, documents, and source content are redacted by default. Content
capture requires the explicit `EVAL_TRACKING_CAPTURE_CONTENT=true` opt-in and
should only be used with non-sensitive fixtures. The opt-in is not a promise
that arbitrary user data is safe to publish.

## Consequences

- Offline evaluation remains dependency-free with no tracking server.
- A local MLflow file store is sufficient for discovery and inspection; a
  server is optional and not required by CI.
- The mirror can be incomplete during outages, so reproducibility and review
  must use the local JSON artifact, never MLflow as the source of truth.
- Retention and filesystem permissions for the tracking URI are the operator's
  responsibility.

## Related

[Reproducible evaluation runs](../reproducibility.md),
[Architecture](../architecture.md), [ADR 015](015-canonical-benchmark-matrix.md),
[ADR 013](013-versioned-benchmark-results.md)
