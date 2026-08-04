from __future__ import annotations

import json
from pathlib import Path

import pytest

from localrag.settings import (
    ConfigError,
    Settings,
    get_settings,
    load_settings,
    set_current_settings,
)


def test_default_chunk_overlap_is_within_10_to_20_percent_of_max_chars() -> None:
    settings = Settings()
    ratio = settings.chunk_overlap_chars / settings.chunk_max_chars
    assert 0.10 <= ratio <= 0.20


def test_configuration_precedence_is_defaults_yaml_dotenv_environment_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("RAG_TOP_K=4\nCHROMA_COLLECTION_NAME=dotenv\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "retrieval:\n  top_k: 3\n  mode: vector\nchroma_collection_name: yaml\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_TOP_K", "5")

    settings = load_settings(config, cli_overrides={"rag_top_k": 6})

    assert settings.rag_top_k == 6
    assert settings.chroma_collection_name == "dotenv"
    assert settings.retrieval_mode == "vector"


def test_yaml_sections_and_environment_interpolation_resolve_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "embedding:\n  provider: ${EMBED_PROVIDER}\n  model: local-model\n"
        "  timeout_seconds: 33\n"
        "dataset:\n  roots: [documents]\n"
        "audit_log_path: logs/audit.jsonl\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")

    settings = load_settings(config)

    assert settings.embedding_provider == "ollama"
    assert settings.embedding_timeout_seconds == 33
    assert settings.audit_log_path == str(tmp_path / "logs/audit.jsonl")


def test_missing_malformed_and_unknown_yaml_fail_actionably(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(tmp_path / "missing.yaml")

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("retrieval: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_settings(malformed)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("retrieval:\n  top_k: 3\n  typo: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"retrieval\.typo"):
        load_settings(unknown)


def test_snapshot_redacts_secrets_and_host_paths(tmp_path: Path) -> None:
    settings = load_settings(
        None,
        cli_overrides={
            "api_key": "secret-api-key",
            "openai_api_key": "secret-openai-key",
            "chroma_persist_path": str(tmp_path / "private"),
        },
    )

    snapshot = settings.resolved_snapshot()

    assert snapshot["api_key"] == "<redacted>"
    assert snapshot["openai_api_key"] == "<redacted>"
    assert str(tmp_path) not in json.dumps(snapshot)
    assert snapshot["chroma_persist_path"] == "<path>"


def test_legacy_yaml_alias_warns_and_cross_field_validation_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("ollama:\n  embed_model: old-model\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "environment-model")

    with pytest.warns(DeprecationWarning, match="ollama\\.embed_model"):
        settings = load_settings(config)
    assert settings.ollama_embed_model == "environment-model"

    with pytest.raises(ValueError, match="chunk_min_chars"):
        load_settings(None, {"chunk_max_chars": 10, "chunk_min_chars": 20})


def test_cached_settings_are_isolated_by_current_execution_context() -> None:
    first = load_settings(None, {"tenant_id": "first"})
    second = load_settings(None, {"tenant_id": "second"})

    set_current_settings(first)
    assert get_settings().tenant_id == "first"
    set_current_settings(second)
    assert get_settings().tenant_id == "second"
