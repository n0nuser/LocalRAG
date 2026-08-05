"""Durable local audit trail for RAG queries (not a regulatory-compliance system)."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localrag import metrics as app_metrics

logger = logging.getLogger(__name__)
_audit_lock = threading.RLock()


def write_audit_record(
    audit_log_path: str,
    *,
    correlation_id: str,
    question: str,
    sources: list[dict[str, Any]],
    answer: str,
    model: str,
    provider: str = "unknown",
    latency_ms: float,
    max_bytes: int = 10_000_000,
    retention_seconds: float = 2_592_000.0,
    metadata_only: bool = False,
    redact_content: bool = False,
) -> None:
    """Append a bounded, optionally content-minimized JSON line audit record."""
    if not audit_log_path:
        return
    if metadata_only:
        question = ""
        sources = []
        answer = ""
    elif redact_content:
        question = f"<redacted:{len(question)} chars>"
        sources = [{"source": "<redacted>"} for _ in sources]
        answer = f"<redacted:{len(answer)} chars>"
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "correlation_id": correlation_id,
        "question": question,
        "sources": sources,
        "answer": answer,
        "model": model,
        "provider": provider,
        "latency_ms": latency_ms,
    }
    try:
        with _audit_lock:
            path = Path(audit_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _expire_audit_files(path, retention_seconds)
            if path.exists() and path.stat().st_size >= max_bytes:
                rotated = path.with_name(f"{path.name}.1")
                rotated.unlink(missing_ok=True)
                path.replace(rotated)
                app_metrics.audit_log_rotations_total.inc()
            encoded = json.dumps(record, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > max_bytes:
                app_metrics.audit_log_oversized_records_total.inc()
                record.update(question="", sources=[], answer="", truncated=True)
                encoded = json.dumps(record, separators=(",", ":"))
                # A JSONL record has unavoidable timestamp and provenance overhead;
                # retain a small practical minimum rather than silently losing it.
                if len(encoded.encode("utf-8")) > max(max_bytes, 256):
                    logger.warning("audit_log_record_too_large path=%s", audit_log_path)
                    app_metrics.audit_log_write_failures_total.inc()
                    return
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
    except OSError:
        app_metrics.audit_log_write_failures_total.inc()
        logger.exception("audit_log_write_failed path=%s", audit_log_path)


def _expire_audit_files(path: Path, retention_seconds: float) -> None:
    if retention_seconds <= 0:
        return
    cutoff = time.time() - retention_seconds
    for candidate in path.parent.glob(f"{path.name}*"):
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
        except OSError:
            app_metrics.audit_log_cleanup_failures_total.inc()
            logger.warning("audit_log_cleanup_failed path=%s", candidate)
