from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.environment import (
    SEED_COVERAGE,
    Capability,
    _cpu_count,
    _gpu_info,
    _model_digest,
    _ollama_model_digests,
    _package_versions,
    _settings_snapshot,
    _uv_lock_sha256,
    capture_run_metadata,
    resolve_seed,
)

# Deterministic record selection (_select_records) is tested against the real
# DatasetRecord type in tests/test_eval_dataset.py, alongside the registry and
# schema it now depends on.

# --- seed precedence ---


def test_resolve_seed_cli_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_SEED", "999")
    seed, source = resolve_seed(cli_seed=7)
    assert (seed, source) == (7, "cli")


def test_resolve_seed_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_SEED", "13")
    seed, source = resolve_seed(cli_seed=None)
    assert (seed, source) == (13, "config")


def test_resolve_seed_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_SEED", raising=False)
    seed, source = resolve_seed(cli_seed=None)
    assert (seed, source) == (42, "default")


def test_resolve_seed_rejects_non_integer_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_SEED", "not-a-number")
    with pytest.raises(ValueError, match="not a valid integer seed"):
        resolve_seed(cli_seed=None)


def test_seed_coverage_documents_every_random_operation() -> None:
    """Each entry must be an explicit bool, not inferred — this is what #83 requires be visible."""
    assert SEED_COVERAGE
    for operation, covered in SEED_COVERAGE.items():
        assert isinstance(operation, str)
        assert isinstance(covered, bool)


# --- model digest resolution ---


def test_model_digest_exact_match() -> None:
    cap = _model_digest({"gemma3:4b": "sha-1"}, "gemma3:4b", ollama_url="http://x")
    assert cap.status == "available"
    assert cap.value == "sha-1"


def test_model_digest_falls_back_to_latest_tag() -> None:
    cap = _model_digest(
        {"nomic-embed-text:latest": "sha-2"}, "nomic-embed-text", ollama_url="http://x"
    )
    assert cap.status == "available"
    assert cap.value == "sha-2"


def test_model_digest_missing_model_is_unavailable_with_reason() -> None:
    cap = _model_digest({"gemma3:4b": "sha-1"}, "llama3", ollama_url="http://x")
    assert cap.status == "unavailable"
    assert cap.value is None
    assert cap.reason


def test_model_digest_unreachable_ollama_is_unavailable_with_reason() -> None:
    cap = _model_digest(None, "gemma3:4b", ollama_url="http://x")
    assert cap.status == "unavailable"
    assert "http://x" in (cap.reason or "")


def test_ollama_model_digests_returns_none_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A down Ollama must degrade to None, not fail the benchmark."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert _ollama_model_digests("http://localhost:11434") is None


def test_ollama_model_digests_skips_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "a", "digest": "d1"}, {"name": "b"}, {"digest": "d3"}]}

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _Resp())
    assert _ollama_model_digests("http://localhost:11434") == {"a": "d1"}


# --- capability semantics ---


def test_capability_ok_is_available() -> None:
    cap = Capability.ok("value")
    assert cap.status == "available"
    assert cap.value == "value"
    assert cap.reason is None


def test_capability_unsupported_has_no_value() -> None:
    cap = Capability.unsupported("no GPU on this host")
    assert cap.status == "unsupported"
    assert cap.value is None
    assert cap.reason == "no GPU on this host"


def test_capability_unavailable_has_no_value() -> None:
    cap = Capability.unavailable("git not installed")
    assert cap.status == "unavailable"
    assert cap.value is None
    assert cap.reason == "git not installed"


def test_gpu_info_never_raises_and_has_a_status() -> None:
    """Whatever this host is (GPU or not), the probe must resolve to a real status, never crash."""
    cap = _gpu_info()
    assert cap.status in ("available", "unsupported")


def test_cpu_count_is_available_on_this_platform() -> None:
    cap = _cpu_count()
    assert cap.status == "available"
    assert isinstance(cap.value, int)
    assert cap.value > 0


# --- run metadata ---


@pytest.fixture
def offline_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.environment._ollama_model_digests", lambda _url: None)


@pytest.mark.usefixtures("offline_ollama")
def test_capture_run_metadata_records_seed_and_models() -> None:
    metadata = capture_run_metadata(
        seed=123,
        seed_source="cli",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    assert metadata.seed == 123
    assert metadata.seed_source == "cli"
    assert metadata.judge_model == "gemma3:4b"
    assert metadata.embedding_model == "nomic-embed-text"


@pytest.mark.usefixtures("offline_ollama")
def test_capture_run_metadata_unreachable_ollama_yields_unavailable_digests() -> None:
    """Never fabricate a digest — an unreachable provider shows unavailable + a reason."""
    metadata = capture_run_metadata(
        seed=1,
        seed_source="default",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    assert metadata.judge_model_digest.status == "unavailable"
    assert metadata.judge_model_digest.value is None
    assert metadata.judge_model_digest.reason
    assert metadata.embedding_model_digest.status == "unavailable"


def test_capture_run_metadata_includes_provenance_fields() -> None:
    """The acceptance criteria for #83: dataset-adjacent identity, git, lock, hardware, config."""
    metadata = capture_run_metadata(
        seed=42,
        seed_source="default",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    payload = metadata.to_dict()

    for key in (
        "metadata_schema_version",
        "seed",
        "seed_source",
        "seed_coverage",
        "git_sha",
        "git_dirty",
        "uv_lock_sha256",
        "package_versions",
        "python_version",
        "platform_summary",
        "cpu_count",
        "total_ram_gb",
        "gpu",
        "judge_model_digest",
        "embedding_model_digest",
        "settings_snapshot",
    ):
        assert key in payload, f"missing provenance key: {key}"

    # This repo is a git checkout with a lockfile, so these must actually resolve.
    assert metadata.git_sha.status == "available", "expected a git SHA in a git checkout"
    assert metadata.uv_lock_sha256.status == "available", "expected a uv.lock hash"
    assert metadata.python_version


def test_capture_run_metadata_never_fabricates_missing_values() -> None:
    """A Capability with status != 'available' must carry value=None, never a guessed value."""
    metadata = capture_run_metadata(
        seed=42,
        seed_source="default",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://unreachable.invalid:11434",
    )
    for cap in (
        metadata.git_sha,
        metadata.git_dirty,
        metadata.uv_lock_sha256,
        metadata.cpu_count,
        metadata.total_ram_gb,
        metadata.gpu,
        metadata.judge_model_digest,
        metadata.embedding_model_digest,
    ):
        if cap.status != "available":
            assert cap.value is None
            assert cap.reason, f"non-available capability missing a reason: {cap}"


def test_run_metadata_is_json_serializable() -> None:
    """Metadata is embedded in the results JSON, so it must serialize cleanly."""
    metadata = capture_run_metadata(
        seed=42,
        seed_source="default",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    round_tripped = json.loads(json.dumps(metadata.to_dict()))
    assert round_tripped["seed"] == 42
    assert round_tripped["git_sha"]["status"] in ("available", "unavailable")


def test_metadata_schema_version_is_recorded() -> None:
    metadata = capture_run_metadata(
        seed=1,
        seed_source="default",
        judge_model="m",
        embedding_model="e",
        ollama_url="http://x",
    )
    assert metadata.metadata_schema_version >= 1


# --- redaction ---


def test_settings_snapshot_contains_retrieval_knobs() -> None:
    snapshot = _settings_snapshot()
    for key in ("rag_top_k", "retrieval_mode", "chunk_chars", "llm_temperature"):
        assert key in snapshot, f"missing settings key: {key}"


def test_settings_snapshot_excludes_secrets_and_paths() -> None:
    """The snapshot is an allowlist — host paths and credentials must not leak in."""
    snapshot = _settings_snapshot()
    for key in (
        "api_key",
        "anthropic_api_key",
        "openai_api_key",
        "chroma_persist_path",
        "upload_dir",
        "audit_log_path",
    ):
        assert key not in snapshot


def test_full_metadata_payload_contains_no_secret_field_names() -> None:
    """End-to-end redaction check on the exact payload written to evals/results/*.json."""
    metadata = capture_run_metadata(
        seed=1,
        seed_source="default",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    serialized = json.dumps(metadata.to_dict())
    for forbidden in ("api_key", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "password", "secret"):
        assert forbidden not in serialized, f"leaked secret-like field: {forbidden}"


def test_full_metadata_payload_contains_no_absolute_host_paths() -> None:
    """settings_snapshot is allowlisted to keep host filesystem layout out of results."""
    metadata = capture_run_metadata(
        seed=1,
        seed_source="default",
        judge_model="gemma3:4b",
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    payload = metadata.to_dict()
    assert "chroma_persist_path" not in payload["settings_snapshot"]
    assert "upload_dir" not in payload["settings_snapshot"]


# --- dependency identity ---


def test_uv_lock_hash_is_stable() -> None:
    assert _uv_lock_sha256().value == _uv_lock_sha256().value


def test_uv_lock_hash_missing_file_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.environment.UV_LOCK_PATH", Path("/nonexistent/uv.lock"))
    cap = _uv_lock_sha256()
    assert cap.status == "unavailable"
    assert cap.value is None


def test_package_versions_includes_tracked_packages_actually_installed() -> None:
    versions = _package_versions()
    # ragas and httpx are hard runtime dependencies of the eval suite itself.
    assert "ragas" in versions
    assert "httpx" in versions
    for version in versions.values():
        assert version, "package version string must not be empty"
