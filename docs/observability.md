# Observability

LocalRAG uses OpenTelemetry only when explicitly enabled. The default install
does not import or require telemetry packages, and `OTEL_ENABLED=false` is a
no-op. Install the optional integration with:

```bash
uv sync --locked --extra observability
```

Set `OTEL_EXPORTER_ENDPOINT` to an OTLP/HTTP collector, such as a local
collector at `http://localhost:4318`. `OTEL_SAMPLE_RATE` is a ratio from `0` to
`1`; exporter timeout, bounded retry configuration, and shutdown flush are
configured independently. Exporter initialization, flush, shutdown, and span
creation failures are logged and never change request or ingestion outcomes.

Spans use stable `localrag.*` names for API/CLI queries, ingestion parse/chunk/
embed, vector/BM25/RRF/freshness/rerank/compression/adaptive retrieval,
request ID context; background ingestion jobs copy the submitting context.
Trace IDs are diagnostic context only and do not replace Prometheus metric
labels or audit records.

## Privacy

Attributes are allowlisted, low-cardinality, and capped at
`OTEL_MAX_ATTRIBUTE_LENGTH`. Query text, documents, prompts, answers, tokens,
API keys, credentials, and arbitrary metadata are prohibited by default. Set
`OTEL_CAPTURE_CONTENT=true` only for a trusted local collector and a deliberate
diagnostic session; content remains bounded but may contain sensitive data.

Langfuse is not part of the core tracing path. Its package has a separate
optional extra for a future adapter, and `LANGFUSE_ENABLED` does nothing in this
release. RAGAS and manual evaluation remain authoritative and do not require
telemetry.

## Prometheus Metrics

The `/metrics` endpoint exposes bounded counters and histograms for query
duration, retrieved chunks, generated tokens, query/provider failures, cache
hits and misses, ingestion failures, background job terminal status, upload
cleanup, audit-log rotation, and HTTP failures by status class. Labels are limited to transport/outcome
 operation names and provider identity; model names, question text, source
 paths, and request IDs are never labels. JSON, adaptive, and SSE queries record
 the same duration, retrieval, token, audit, and failure signals. Fallback use,
 upload quota rejection, oversized audit records, and audit write/cleanup
 failures have dedicated bounded counters.

Useful alerts include `rate(localrag_query_failures_total[5m]) > 0`, a sustained
`localrag_ingest_jobs_pending` near the configured cap, and any increase in
`localrag_provider_fallbacks_total` or `localrag_upload_quota_rejections_total`.
