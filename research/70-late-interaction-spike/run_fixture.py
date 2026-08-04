"""Run the dependency-free #70 fixture and emit canonical evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.late_interaction import LateInteractionIndex


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return numerator / (left_norm * right_norm)


def _recall_at_one(ranked: list[str], relevant: list[str]) -> float:
    return float(bool(ranked and ranked[0] in relevant))


def _bm25ish(query: str, text: str) -> int:
    query_tokens = set(query.lower().split())
    return sum(token in text.lower().split() for token in query_tokens)


def run(fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = fixture["documents"]
    queries = fixture["queries"]
    index = LateInteractionIndex()
    tracemalloc.start()
    build_start = time.perf_counter()
    for document in documents:
        index.add(document["id"], document["tokens"])
    build_seconds = time.perf_counter() - build_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    index_path = output_dir / "late-interaction-index.json"
    index.save(index_path)

    quality: dict[str, list[float]] = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
        "late_interaction": [],
    }
    warm_latencies: list[float] = []
    for query in queries:
        dense = sorted(
            ((doc["id"], _cosine(query["dense"], doc["dense"])) for doc in documents),
            key=lambda item: (-item[1], item[0]),
        )
        bm25 = sorted(
            ((doc["id"], _bm25ish(query["text"], doc["text"])) for doc in documents),
            key=lambda item: (-item[1], item[0]),
        )
        dense_scores = dict(dense)
        bm25_scores = dict(bm25)
        hybrid = sorted(
            ((doc["id"], dense_scores[doc["id"]] + bm25_scores[doc["id"]]) for doc in documents),
            key=lambda item: (-item[1], item[0]),
        )
        late = index.search(query["tokens"], top_k=len(documents))
        index.search(query["tokens"], top_k=len(documents))
        for _ in range(20):
            start = time.perf_counter()
            index.search(query["tokens"], top_k=len(documents))
            warm_latencies.append(time.perf_counter() - start)
        relevant = query["relevant"]
        quality["dense"].append(_recall_at_one([item[0] for item in dense], relevant))
        quality["bm25"].append(_recall_at_one([item[0] for item in bm25], relevant))
        quality["hybrid"].append(_recall_at_one([item[0] for item in hybrid], relevant))
        quality["late_interaction"].append(_recall_at_one([item[0] for item in late], relevant))

    result = {
        "schema_version": 1,
        "run_id": "late-interaction-fixture-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": fixture["dataset"],
        "selected_ids": [query["id"] for query in queries],
        "metrics": [
            {
                "descriptor": {"name": f"recall_at_1_{name}", "direction": "higher_is_better"},
                "value": statistics.mean(values),
            }
            for name, values in quality.items()
        ]
        + [
            {
                "descriptor": {
                    "name": "late_index_bytes",
                    "direction": "lower_is_better",
                    "unit": "bytes",
                },
                "value": index_path.stat().st_size,
            },
        ],
        "provenance": {
            "experiment": "issue-70-late-interaction-fixture",
            "fixture": str(fixture_path),
            "model": "deterministic hand-authored token vectors; no model download",
            "tokenizer": "whitespace split in fixture text; token vectors are explicit",
            "hardware": {"platform": platform.platform(), "python": platform.python_version()},
            "precision": "float64 Python values",
            "batch_size": 1,
            "warmup_repetitions": 1,
            "measurement_repetitions": 20,
            "cold_start": (
                "not measured: dependency-free process startup dominates this toy fixture"
            ),
            "peak_python_bytes": peak_bytes,
            "index_build_seconds": build_seconds,
            "warm_query_p50_ms": statistics.median(warm_latencies) * 1000,
            "warm_query_p95_ms": sorted(warm_latencies)[max(0, int(len(warm_latencies) * 0.95) - 1)]
            * 1000,
            "index_artifact_bytes": index_path.stat().st_size,
            "vram": "not measured; CPU-only fixture",
        },
        "status": "complete",
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    revision = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "matrix_id": "late-interaction-fixture",
        "run_id": "late-interaction-fixture-v1",
        "profile": "fixture",
        "dataset": fixture["dataset"],
        "corpus": {
            "identity": fixture["dataset"]["dataset_id"],
            "checksum": fixture["dataset"]["checksum"],
        },
        "code_revision": revision,
        "code_dirty": False,
        "model": {"provider": "none", "embedding": "deterministic-fixture", "generation": "none"},
        "effective_config": {"top_k": 1, "seed": 42, "precision": "float64"},
        "mode": "fixture-offline",
        "supported_dimensions": {"retrieval_mode": ("dense", "bm25", "hybrid", "late_interaction")},
        "seed": 42,
        "status": "complete",
        "started_at": result["timestamp"],
        "finished_at": datetime.now(UTC).isoformat(),
        "cases": [],
        "artifact_paths": {"result": str(result_path), "index": str(index_path)},
        "exit_code": 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("fixture.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("evals/results/late-interaction-fixture")
    )
    args = parser.parse_args()
    run(args.fixture, args.output)
