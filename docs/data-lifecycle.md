# Data Lifecycle

`POST /ingest/upload` treats multipart files as temporary ingest artifacts, not
durable sources. Files are streamed with `UPLOAD_MAX_BYTES`, stored under
`UPLOAD_DIR` using a SHA-256 content-addressed name, and removed after ingest by
default. Set `UPLOAD_RETENTION_SECONDS` to retain artifacts for rebuilds;
`UPLOAD_QUOTA_BYTES` removes the oldest retained files past its bound. Failed
and partial uploads are removed as well.

The query audit log is disabled unless `AUDIT_LOG_PATH` is set. It rotates at
`AUDIT_LOG_MAX_BYTES` and removes old log/rotation files after
`AUDIT_LOG_RETENTION_SECONDS`. `AUDIT_LOG_METADATA_ONLY=true` stores only
correlation ID, model, and latency. `AUDIT_LOG_REDACT_CONTENT=true` keeps
content lengths while replacing questions, answers, and source values.
