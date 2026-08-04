from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from localrag.cli.app import app
from localrag.cli.commands import benchmark as benchmark_command
from localrag.cli.commands import inspect as inspect_command

runner = CliRunner()


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]], metadata: Any = None) -> None:
        self.rows = rows
        self.metadata = metadata

    def count(self) -> int:
        return len(self.rows)

    def get(self, **_: Any) -> dict[str, Any]:
        return {
            "ids": [row["id"] for row in self.rows],
            "documents": [row["document"] for row in self.rows],
            "metadatas": [row["metadata"] for row in self.rows],
        }


def test_help_lists_benchmark_and_inspect_contracts() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "--collection" in result.stdout
    assert "--sample-count" in result.stdout
    assert "--format" in result.stdout

    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout
    assert "--config" in result.stdout
    assert "--matrix" in result.stdout
    assert "--result-output" in result.stdout


def test_inspect_json_is_sorted_bounded_and_sanitized(monkeypatch: Any) -> None:
    collection = FakeCollection(
        [
            {"id": "b", "document": "second", "metadata": {"source": "b.txt"}},
            {"id": "a", "document": "A\x00" + "x" * 100, "metadata": {"source": "a.txt"}},
        ],
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(inspect_command, "open_collection", lambda _: collection)

    result = runner.invoke(
        app,
        [
            "inspect",
            "--collection",
            "docs",
            "--sample-count",
            "1",
            "--max-chars",
            "20",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["vector_count"] == 2
    assert len(payload["samples"]) == 1
    assert payload["samples"][0]["id"] == "a"
    assert "\x00" not in payload["samples"][0]["document"]
    assert len(payload["samples"][0]["document"]) <= 20


def test_inspect_missing_collection_has_exit_code_two(monkeypatch: Any) -> None:
    def missing(_: str) -> Any:
        raise inspect_command.CollectionNotFoundError("collection 'missing' does not exist")

    monkeypatch.setattr(inspect_command, "open_collection", missing)
    result = runner.invoke(app, ["inspect", "--collection", "missing"])
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_inspect_empty_collection_is_successful(monkeypatch: Any) -> None:
    monkeypatch.setattr(inspect_command, "open_collection", lambda _: FakeCollection([]))
    result = runner.invoke(app, ["inspect", "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["samples"] == []


def test_benchmark_delegates_to_matrix_runner(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[Any, Path, bool]] = []

    def fake_runner(config: Any, output: Path, *, dry_run: bool = False) -> dict[str, Any]:
        calls.append((config, output, dry_run))
        return {
            "exit_code": 0,
            "artifact_paths": {"matrix": "artifact"},
            "status": "complete",
            "cases": [],
        }

    monkeypatch.setattr(benchmark_command, "run_matrix", fake_runner)
    result = runner.invoke(
        app,
        ["benchmark", "--dataset", "localrag-core", "--result-output", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert calls[0][1] == tmp_path
    assert calls[0][2] is True
    assert '"artifact_paths"' in result.stdout


def test_benchmark_malformed_runner_result_fails(monkeypatch: Any) -> None:
    def malformed_runner(*_: Any, **__: Any) -> dict[str, Any]:
        return {"status": "complete"}

    monkeypatch.setattr(benchmark_command, "run_matrix", malformed_runner)
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 1
    assert "malformed" in result.stderr
