from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.leaderboard import LeaderboardError, generate_leaderboard


def _artifact(path: Path, *, model: str = "model-b", dataset: str = "core") -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_kind": "matrix_case",
        "artifact_id": f"{model}-artifact",
        "source_artifact": "canonical.json",
        "dataset": {"id": dataset, "version": "1.0", "split": "test", "hash": "sha-dataset"},
        "evaluation_schema_version": 1,
        "model": {
            "provider": "ollama",
            "name": model,
            "revision": "rev-1",
            "digest": "sha-model",
            "quantization": "Q4",
            "runtime": "ollama",
        },
        "embedding": {"name": "embed", "revision": "rev-e", "digest": "sha-embed"},
        "retrieval": {"mode": "hybrid", "config": {"top_k": 5}},
        "reranker": {"id": "none", "version": "1", "enabled": False},
        "chunker": {"id": "structural", "version": "1", "config": {"max_chars": 1200}},
        "hardware": {"os": "Linux", "cpu": "x86", "gpu": "none", "ram_gb": 16},
        "temperature": "warm",
        "repetitions": 3,
        "warmup_repetitions": 1,
        "metrics": {
            "quality": {
                "definition": "mean answer quality",
                "unit": "score",
                "mean": 0.8,
                "median": 0.8,
                "p95": 0.81,
                "stddev": 0.01,
            }
        },
        "latency": {
            "unit": "ms",
            "method": "monotonic wall clock per query",
            "mean": 100,
            "median": 100,
            "p95": 110,
            "stddev": 5,
        },
        "resources": {"unit": "MiB", "method": "peak RSS sampled by /proc", "peak": 512},
        "run": {
            "timestamp": "2026-08-04T00:00:00Z",
            "code_revision": "sha-code",
            "dependency_lock_hash": "sha-lock",
            "seed": 42,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_leaderboard_is_deterministic_and_orders_rows(tmp_path: Path) -> None:
    first = _artifact(tmp_path / "b.json", model="model-b")
    second = _artifact(tmp_path / "a.json", model="model-a")
    result_a = generate_leaderboard([first, second], tmp_path / "one.md", tmp_path / "one.json")
    result_b = generate_leaderboard([second, first], tmp_path / "two.md", tmp_path / "two.json")

    assert result_a.row_count == result_b.row_count == 2
    assert (tmp_path / "one.md").read_text() == (tmp_path / "two.md").read_text()
    assert (tmp_path / "one.json").read_text() == (tmp_path / "two.json").read_text()
    assert "model-a" in (tmp_path / "one.md").read_text()
    assert (tmp_path / "one.md").read_text().index("model-a") < (
        tmp_path / "one.md"
    ).read_text().index("model-b")


def test_empty_input_is_informative(tmp_path: Path) -> None:
    result = generate_leaderboard([], tmp_path / "empty.md")
    assert result.row_count == 0
    assert "No valid benchmark artifacts" in (tmp_path / "empty.md").read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": {"id": "other", "version": "1.0", "split": "test", "hash": "different"}},
        {"temperature": "cold"},
    ],
)
def test_incompatible_rows_fail(tmp_path: Path, change: dict[str, Any]) -> None:
    first = _artifact(tmp_path / "first.json")
    second = _artifact(tmp_path / "second.json", model="model-c")
    payload = json.loads(second.read_text())
    payload.update(change)
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LeaderboardError, match="incompatible"):
        generate_leaderboard([first, second], tmp_path / "out.md")


def test_missing_exact_matrix_row_fails(tmp_path: Path) -> None:
    source = _artifact(tmp_path / "source.json", model="model-a")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identities": [
                    {
                        "provider": "ollama",
                        "name": "model-a",
                        "revision": "rev-1",
                        "digest": "sha-model",
                    },
                    {
                        "provider": "ollama",
                        "name": "model-b",
                        "revision": "rev-1",
                        "digest": "sha-model",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LeaderboardError, match="missing"):
        generate_leaderboard([source], tmp_path / "out.md", matrix=matrix)


def test_malformed_and_incomplete_artifacts_fail(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(LeaderboardError, match=r"bad\.json"):
        generate_leaderboard([bad], tmp_path / "out.md")

    incomplete = _artifact(tmp_path / "incomplete.json")
    payload = json.loads(incomplete.read_text())
    del payload["hardware"]
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LeaderboardError, match="hardware"):
        generate_leaderboard([incomplete], tmp_path / "out.md")
