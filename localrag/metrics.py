"""Prometheus metrics registry for LocalRAG.

Import the singletons from here; never instantiate metric objects elsewhere.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

query_duration_seconds = Histogram(
    "localrag_query_duration_seconds",
    "Wall-clock seconds for a full JSON query (retrieval + generation).",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

tokens_used_total = Counter(
    "localrag_tokens_used_total",
    "Total LLM tokens consumed (approximated by token stream length).",
    labelnames=["model"],
)

chunks_retrieved_total = Counter(
    "localrag_chunks_retrieved_total",
    "Total vector-store chunks retrieved across all queries.",
)

ingested_documents_total = Counter(
    "localrag_ingested_documents_total",
    "Total documents successfully ingested.",
)

query_failures_total = Counter(
    "localrag_query_failures_total", "Queries that failed before producing a response."
)
provider_failures_total = Counter(
    "localrag_provider_failures_total", "Provider calls that failed.", labelnames=["provider"]
)
cache_operations_total = Counter(
    "localrag_query_cache_operations_total", "Query cache outcomes.", labelnames=["operation"]
)
ingest_failures_total = Counter("localrag_ingest_failures_total", "Sources that failed ingestion.")
ingest_jobs_total = Counter(
    "localrag_ingest_jobs_total",
    "Background ingest jobs by terminal status.",
    labelnames=["status"],
)

upload_cleanup_total = Counter(
    "localrag_upload_cleanup_total",
    "Uploaded temporary artifacts removed by retention or quota cleanup.",
    labelnames=["reason"],
)

audit_log_rotations_total = Counter(
    "localrag_audit_log_rotations_total",
    "Audit log files rotated after reaching the configured size limit.",
)
