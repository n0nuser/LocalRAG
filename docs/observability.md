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
