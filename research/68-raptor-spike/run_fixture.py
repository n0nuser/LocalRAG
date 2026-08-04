"""Run the bounded RAPTOR fixture and emit #73/#84-shaped evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_68_raptor import raptor


def run(fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = raptor.RaptorConfig(cluster_count=2, seed=42, context_max_chars=2400)
    chunks = [
        raptor.LeafChunk.create(
            item["source_id"], item["chunk_id"], item["text"], tuple(item["embedding"])
        )
        for item in fixture["chunks"]
    ]
    embed = {item["text"]: tuple(item["embedding"]) for item in fixture["queries"]}
    build_start = time.perf_counter()
    tracemalloc.start()
    artifact = raptor.RaptorBuilder(
        config, fixture["embedding_model"], fixture["summarizer_model"]
    ).build(chunks)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    build_seconds = time.perf_counter() - build_start
    artifact_path = output_dir / "artifact.json"
    artifact.save_atomic(artifact_path)
    retriever = raptor.RaptorRetriever(artifact, lambda query: embed[query])
    latencies: list[float] = []
    hits = 0
    for query in fixture["queries"]:
        for _ in range(20):
            start = time.perf_counter()
            result = retriever.search(query["text"], top_k=3)
            latencies.append(time.perf_counter() - start)
        hits += int(any(query["relevant_source_id"] in node.source_ids for node in result))
    result_path = output_dir / "result.json"
    result = {
        "schema_version": 1,
        "run_id": "raptor-fixture-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {"dataset_id": fixture["dataset_id"], "version": fixture["version"]},
        "selected_ids": [query["id"] for query in fixture["queries"]],
        "metrics": [
            {
                "descriptor": {"name": "source_recall_at_3", "direction": "higher_is_better"},
                "value": hits / len(fixture["queries"]),
            },
            {
                "descriptor": {
                    "name": "artifact_bytes",
                    "direction": "lower_is_better",
                    "unit": "bytes",
                },
                "value": artifact_path.stat().st_size,
            },
        ],
        "provenance": {
            "experiment": "issue-68-raptor-fixture",
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "embedding_model": fixture["embedding_model"],
            "summarizer_model": fixture["summarizer_model"],
            "config": artifact.config.__dict__,
            "hardware": {"platform": platform.platform(), "python": platform.python_version()},
            "cold_start": "not measured: dependency-free fixture process startup dominates",
            "warmup_repetitions": 0,
            "measurement_repetitions": 20,
            "build_seconds": build_seconds,
            "warm_query_p50_ms": statistics.median(latencies) * 1000,
            "warm_query_p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] * 1000,
            "peak_python_bytes": peak_bytes,
            "llm_calls": artifact.build_stats["summary_calls"],
            "summary_failures": artifact.build_stats["failed_summaries"],
        },
        "status": "complete",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "matrix_id": "raptor-fixture",
        "run_id": result["run_id"],
        "profile": "fixture",
        "dataset": result["dataset"],
        "mode": "fixture-offline",
        "seed": 42,
        "status": "complete",
        "artifact_paths": {"result": str(result_path), "artifact": str(artifact_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    run(Path(__file__).with_name("fixture.json"), Path("research/68-raptor-spike/artifacts"))
