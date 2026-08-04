# ADR 030: Optional OpenTelemetry Observability Boundary

## Status

Accepted

## Decision

LocalRAG instruments operational stages with OpenTelemetry spans behind the
`observability` extra and `OTEL_ENABLED`. The base installation and disabled
configuration remain no-op and offline-first. OpenTelemetry is the vendor-
neutral boundary; Langfuse, if added later, must remain a separate optional
adapter and may not be coupled to RAG, ingestion, or evaluation contracts.

Span names are stable and hierarchical across API/CLI query, ingestion,
retrieval, generation, and benchmark/evaluation stages. Attributes are
allowlisted, bounded, and operational. Query/document/prompt/answer/token
content, credentials, and API keys are prohibited unless the explicit bounded
content opt-in is enabled. Sampling happens in the SDK after this redaction
policy is applied.

Exporter and lifecycle failures are fail-open: they are logged and do not alter
request, ingestion, or benchmark results. API request context and in-process
background jobs preserve trace parentage where available. Traces complement,
but do not duplicate, Prometheus metrics or the local audit record.

## Consequences

Operators can use a local OTLP collector without adding a hosted service or
secrets to the default deployment. Telemetry is intentionally less detailed
than application data, so debugging content-specific failures still uses the
existing local logs/audit controls. Langfuse setup is intentionally deferred
until an adapter contract can be reviewed independently.
