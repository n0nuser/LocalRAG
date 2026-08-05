"""Run the offline GraphRAG fixture and emit #73/#84-shaped evidence."""

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

from research.graphrag_spike import graphrag


def fixture_provider(items: list[graphrag.SourceChunk]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for item in items:
        entities = (
            [
                {"name": "Ada", "entity_type": "person", "confidence": 0.9},
                {"name": "Acme", "entity_type": "organization", "confidence": 0.9},
            ]
            if "Ada" in item.text
            else [{"name": "satellite", "entity_type": "object", "confidence": 0.8}]
        )
        relations = (
            [{"source": "Ada", "target": "Acme", "predicate": "works_at", "confidence": 0.8}]
            if "works" in item.text or "employs" in item.text
            else []
        )
        records[item.chunk_id] = {
            "chunk_id": item.chunk_id,
            "entities": entities,
            "relations": relations,
        }
    return records


def run(fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = [
        graphrag.SourceChunk(item["source_id"], item["chunk_id"], item["text"], item["citation_id"])
        for item in fixture["chunks"]
    ]
    config = graphrag.GraphConfig(max_hops=2, context_max_chars=2400)  # type: ignore[attr-defined]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tracemalloc.start()
    extracted, failed = graphrag.ExtractionRunner(fixture_provider).run(chunks, config)
    artifact = graphrag.build_graph(
        chunks, extracted, config, extractor_identity=fixture["extractor_model"], quarantined=failed
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    build_seconds = time.perf_counter() - started
    artifact_path = output_dir / "artifact.json"
    artifact.save_atomic(artifact_path)
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    query_latencies: list[float] = []
    hits = 0
    for query in fixture["queries"]:
        for _ in range(20):
            query_started = time.perf_counter()
            graph_hits = graphrag.retrieve_graph(artifact, set(query["entities"]))
            graphrag.compose_retrieval(graph_hits, [], by_chunk)
            query_latencies.append(time.perf_counter() - query_started)
        citations = {citation for node in graph_hits for citation in node.provenance.citation_ids}
        hits += int(bool(citations & set(query["relevant_citation_ids"])))
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "graphrag-fixture-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {"dataset_id": fixture["dataset_id"], "version": fixture["version"]},
        "selected_ids": [query["id"] for query in fixture["queries"]],
        "metrics": [
            {
                "descriptor": {"name": "citation_recall_at_3", "direction": "higher_is_better"},
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
            "experiment": "issue-67-graphrag-fixture",
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "embedding_model": fixture["embedding_model"],
            "extractor_model": fixture["extractor_model"],
            "config": asdict_config(config),
            "hardware": {"platform": platform.platform(), "python": platform.python_version()},
            "measurement_repetitions": 20,
            "build_seconds": build_seconds,
            "warm_query_p50_ms": statistics.median(query_latencies) * 1000,
            "warm_query_p95_ms": sorted(query_latencies)[
                max(0, int(len(query_latencies) * 0.95) - 1)
            ]
            * 1000,
            "peak_python_bytes": peak_bytes,
            "extraction_failures": len(failed),
        },
        "status": "complete",
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matrix_id": "graphrag-fixture",
                "run_id": result["run_id"],
                "profile": "fixture",
                "dataset": result["dataset"],
                "mode": "fixture-offline",
                "seed": 42,
                "status": "complete",
                "artifact_paths": {"result": str(result_path), "artifact": str(artifact_path)},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result


def asdict_config(config: graphrag.GraphConfig) -> dict[str, object]:
    return {
        "max_chars_per_chunk": config.max_chars_per_chunk,
        "max_neighbors": config.max_neighbors,
        "max_hops": config.max_hops,
        "context_max_chars": config.context_max_chars,
        "prompt_version": config.prompt_version,
    }


if __name__ == "__main__":
    run(Path(__file__).with_name("fixture.json"), Path(__file__).with_name("artifacts"))
