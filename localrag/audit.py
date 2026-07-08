"""Durable local audit trail for RAG queries (not a regulatory-compliance system)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_audit_record(
    audit_log_path: str,
    *,
    correlation_id: str,
    question: str,
    sources: list[dict[str, Any]],
    answer: str,
    model: str,
    latency_ms: float,
) -> None:
    """Append one JSON line to ``audit_log_path``. No-op when the path is empty."""
    if not audit_log_path:
        return
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "correlation_id": correlation_id,
        "question": question,
        "sources": sources,
        "answer": answer,
        "model": model,
        "latency_ms": latency_ms,
    }
    try:
        path = Path(audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("audit_log_write_failed path=%s", audit_log_path)
