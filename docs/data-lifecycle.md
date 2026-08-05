# Data Lifecycle

`POST /ingest/upload` treats multipart files as temporary ingest artifacts, not
durable sources. Files are streamed with `UPLOAD_MAX_BYTES`, stored under
`UPLOAD_DIR` using a SHA-256 content-addressed name, and removed after ingest by
default. Set `UPLOAD_RETENTION_SECONDS` to retain artifacts for rebuilds;
`UPLOAD_QUOTA_BYTES` removes the oldest retained files past its bound. Failed
and partial uploads are removed as well. Cleanup runs before and after each
upload; quota rejections and cleanup activity are exposed through Prometheus
counters.

The query audit log is disabled unless `AUDIT_LOG_PATH` is set. It rotates at
`AUDIT_LOG_MAX_BYTES` and removes old log/rotation files after
`AUDIT_LOG_RETENTION_SECONDS`. `AUDIT_LOG_METADATA_ONLY=true` stores only
correlation ID, model, and latency. `AUDIT_LOG_REDACT_CONTENT=true` keeps
content lengths while replacing questions, answers, and source values. Oversized
records are reduced to provenance and metadata, or discarded if even that
minimum cannot fit. Write, rotation, oversize, and retention-cleanup outcomes
are metered.

## Ingestion consistency

Ingestion, collection rebuilds, source deletion, and vector queries are serialized
within an API process. Replacing a source writes the new chunks before removing
obsolete chunks and restores the previous source version if a Chroma operation
fails. The BM25 index is rebuilt privately and published as one immutable snapshot,
so queries see a complete old or new corpus rather than a partially refreshed one.
This is a process-local guarantee; sharing one persistent Chroma path between
multiple API processes is outside the supported write model. See [ADR 035](adr/035-atomic-ingestion-replacement.md).
