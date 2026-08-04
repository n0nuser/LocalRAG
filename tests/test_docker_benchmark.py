from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.docker_benchmark import (
    BenchmarkConfigError,
    ModelLock,
    ModelPin,
    _fixture_result,
    load_model_lock,
    reset_results,
    validate_model_lock,
    verify_ollama_models,
)


def test_model_lock_requires_immutable_ollama_digests(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"models": [{"name": "demo", "digest": "sha256:" + "a" * 64}]}),
        encoding="utf-8",
    )

    lock = load_model_lock(path)

    assert validate_model_lock(lock, {"demo"}) == {"demo": "sha256:" + "a" * 64}


def test_model_lock_rejects_tags_and_missing_models(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"models": [{"name": "demo", "digest": "latest"}]}), encoding="utf-8"
    )

    with pytest.raises(BenchmarkConfigError, match="immutable digest"):
        validate_model_lock(load_model_lock(path), {"demo"})
    with pytest.raises(BenchmarkConfigError, match="missing"):
        validate_model_lock(load_model_lock(path), {"other"})


def test_reset_results_removes_stale_exports_but_keeps_directory(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "stale.json").write_text("{}", encoding="utf-8")
    (output / "nested").mkdir()
    (output / "nested" / "artifact.txt").write_text("old", encoding="utf-8")

    reset_results(output)

    assert output.is_dir()
    assert not list(output.iterdir())


def test_model_verification_accepts_ollama_implicit_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "sha256:" + "b" * 64
    lock = ModelLock(models=[ModelPin(name="demo", digest=digest)])
    monkeypatch.setattr(
        "scripts.docker_benchmark._request_json",
        lambda _url: {"models": [{"name": "demo:latest", "digest": digest}]},
    )

    verify_ollama_models("http://ollama", {"demo"}, lock)


def test_model_verification_fails_on_digest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = load_model_lock(Path(__file__).parents[1] / "docker" / "models.lock.json")
    monkeypatch.setattr(
        "scripts.docker_benchmark._request_json",
        lambda _url: {"models": [{"name": "gemma3:4b", "digest": "sha256:" + "c" * 64}]},
    )

    with pytest.raises(BenchmarkConfigError, match="digest mismatch"):
        verify_ollama_models("http://ollama", {"gemma3:4b"}, lock)


def test_fixture_run_exports_a_schema_valid_result(tmp_path: Path) -> None:
    result = _fixture_result(tmp_path, profile="smoke", seed=42)

    assert result.status == "complete"
    assert result.selected_ids
    assert (tmp_path / "result.json").exists()
