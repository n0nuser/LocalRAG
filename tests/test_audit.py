from __future__ import annotations

import json
from pathlib import Path

from localrag.audit import write_audit_record


def test_write_audit_record_appends_json_line(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"

    write_audit_record(
        str(log_path),
        correlation_id="rid-1",
        question="What is X?",
        sources=[{"source": "a.md", "chunk_index": 0}],
        answer="X is Y.",
        model="llama3.2",
        latency_ms=123.4,
    )
    write_audit_record(
        str(log_path),
        correlation_id="rid-2",
        question="Second question",
        sources=[],
        answer="Second answer",
        model="llama3.2",
        latency_ms=50.0,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["correlation_id"] == "rid-1"
    assert first["question"] == "What is X?"
    assert first["answer"] == "X is Y."


def test_write_audit_record_noop_when_path_empty(tmp_path: Path) -> None:
    write_audit_record(
        "",
        correlation_id="rid",
        question="q",
        sources=[],
        answer="a",
        model="m",
        latency_ms=1.0,
    )
    assert list(tmp_path.iterdir()) == []


def test_write_audit_record_metadata_only_and_rotation(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    write_audit_record(
        str(log_path),
        correlation_id="rid",
        question="secret question",
        sources=[{"source": "private.md"}],
        answer="secret answer",
        model="m",
        latency_ms=1,
        max_bytes=1,
        metadata_only=True,
        retention_seconds=0,
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["correlation_id"] == "rid"
    assert record["question"] == ""
    assert record["sources"] == []
    assert record["answer"] == ""


def test_write_audit_record_reduces_oversized_content(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    write_audit_record(
        str(log_path),
        correlation_id="rid",
        question="q" * 10_000,
        sources=[{"source": "s" * 10_000}],
        answer="a" * 10_000,
        model="m",
        latency_ms=1,
        max_bytes=1_000,
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["truncated"] is True
    assert record["question"] == ""
    assert record["answer"] == ""
