from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from localrag.cli.app import app
from localrag.cli.commands import benchmark as benchmark_command
from localrag.cli.commands import inspect as inspect_command

runner = CliRunner()


def plain_help(result: Any) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]], metadata: Any = None) -> None:
        self.rows = rows
        self.metadata = metadata

    def count(self) -> int:
        return len(self.rows)

    def get(
        self,
        include: list[str] | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        # Chroma rejects anything outside this set, and returns ids regardless.
        # Enforcing it here is what makes an invalid `include` fail the suite.
        valid = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}
        invalid = sorted(set(include or []) - valid)
        if invalid:
            message = (
                f"Expected include item to be one of {', '.join(sorted(valid))}, "
                f"got {invalid[0]} in get."
            )
            raise ValueError(message)
        rows = self.rows if limit is None else self.rows[:limit]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }


def test_help_lists_benchmark_and_inspect_contracts() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    help_text = plain_help(result)
    assert "--collection" in help_text
    assert "--sample-count" in help_text
    assert "--format" in help_text

    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    help_text = plain_help(result)
    assert "--dataset" in help_text
    assert "--config" in help_text
    assert "--matrix" in help_text
    assert "--result-output" in help_text


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


def test_leaderboard_help_and_empty_publication(tmp_path: Path) -> None:
    result = runner.invoke(app, ["leaderboard", "--help"])
    assert result.exit_code == 0
    assert "--json-output" in plain_help(result)

    output = tmp_path / "leaderboard.md"
    result = runner.invoke(app, ["leaderboard", "--output", str(output)])
    assert result.exit_code == 0
    assert "No valid benchmark artifacts" in output.read_text(encoding="utf-8")
