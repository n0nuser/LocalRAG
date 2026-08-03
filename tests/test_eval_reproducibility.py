from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.environment import (
    RunMetadata,
    _lookup_digest,
    _ollama_model_digests,
    _settings_snapshot,
    _uv_lock_sha256,
    capture_run_metadata,
)

# Deterministic record selection (_select_records) is tested against the real
# DatasetRecord type in tests/test_eval_dataset.py, alongside the registry and
# schema it now depends on.

# --- model digest resolution ---


def test_lookup_digest_exact_match() -> None:
    assert _lookup_digest({"gemma3:4b": "sha-1"}, "gemma3:4b") == "sha-1"


def test_lookup_digest_falls_back_to_latest_tag() -> None:
    """A bare model name should resolve against Ollama's implicit ``:latest`` tag."""
    assert _lookup_digest({"nomic-embed-text:latest": "sha-2"}, "nomic-embed-text") == "sha-2"


def test_lookup_digest_missing_model_is_none() -> None:
    assert _lookup_digest({"gemma3:4b": "sha-1"}, "llama3") is None


def test_ollama_model_digests_returns_empty_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A down Ollama must degrade to empty metadata, not fail the benchmark."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert _ollama_model_digests("http://localhost:11434") == {}


def test_ollama_model_digests_skips_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "a", "digest": "d1"}, {"name": "b"}, {"digest": "d3"}]}

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _Resp())
    assert _ollama_model_digests("http://localhost:11434") == {"a": "d1"}


# --- run metadata ---


@pytest.fixture
def offline_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.environment._ollama_model_digests", lambda _url: {})


@pytest.mark.usefixtures("offline_ollama")
def test_capture_run_metadata_records_seed_and_models() -> None:
    metadata = capture_run_metadata(
        seed=123,
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    assert metadata.seed == 123
    assert metadata.judge_model == "gemma3:4b"
    assert metadata.embedding_model == "nomic-embed-text"


@pytest.mark.usefixtures("offline_ollama")
def test_capture_run_metadata_includes_provenance_fields() -> None:
    """The acceptance criteria for #83: git SHA, lock hash, hardware, settings."""
    metadata = capture_run_metadata(
        seed=42,
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    payload = metadata.to_dict()

    for key in (
        "git_sha",
        "uv_lock_sha256",
        "python_version",
        "platform_summary",
        "cpu_count",
        "settings_snapshot",
    ):
        assert key in payload, f"missing provenance key: {key}"

    # This repo is a git checkout with a lockfile, so these must actually resolve.
    assert metadata.git_sha, "expected a git SHA in a git checkout"
    assert metadata.uv_lock_sha256, "expected a uv.lock hash"
    assert metadata.python_version


@pytest.mark.usefixtures("offline_ollama")
def test_run_metadata_is_json_serializable() -> None:
    """Metadata is embedded in the results JSON, so it must serialize cleanly."""
    metadata = capture_run_metadata(
        seed=42,
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    round_tripped = json.loads(json.dumps(metadata.to_dict()))
    assert round_tripped["seed"] == 42


def test_settings_snapshot_contains_retrieval_knobs() -> None:
    snapshot = _settings_snapshot()
    for key in ("rag_top_k", "retrieval_mode", "chunk_chars", "llm_temperature"):
        assert key in snapshot, f"missing settings key: {key}"


def test_settings_snapshot_excludes_secrets_and_paths() -> None:
    """The snapshot is an allowlist — host paths and credentials must not leak in."""
    snapshot = _settings_snapshot()
    for key in ("api_key", "anthropic_api_key", "openai_api_key", "chroma_persist_path"):
        assert key not in snapshot


def test_uv_lock_hash_is_stable() -> None:
    assert _uv_lock_sha256() == _uv_lock_sha256()


def test_uv_lock_hash_missing_file_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.environment.UV_LOCK_PATH", Path("/nonexistent/uv.lock"))
    assert _uv_lock_sha256() is None


def test_metadata_dataclass_has_no_required_runtime_deps() -> None:
    """RunMetadata should be constructible directly (used by tests and future tooling)."""
    metadata = RunMetadata(
        seed=1,
        git_sha=None,
        git_dirty=None,
        uv_lock_sha256=None,
        python_version="3.13.0",
        platform_summary="linux",
        cpu_count=None,
        total_ram_gb=None,
        judge_model="m",
        judge_model_digest=None,
        embedding_model="e",
        embedding_model_digest=None,
    )
    assert metadata.to_dict()["settings_snapshot"] == {}
