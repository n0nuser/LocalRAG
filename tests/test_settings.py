from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from localrag.settings import (
    ConfigError,
    Settings,
    get_settings,
    load_settings,
    set_current_settings,
)
from localrag.settings_map import FLAT_TO_PATH, PATH_TO_FLAT, UNGROUPED_FIELDS


def test_default_chunk_overlap_is_within_10_to_20_percent_of_max_chars() -> None:
    settings = Settings()
    ratio = settings.chunk_overlap_chars / settings.chunk_max_chars
    assert 0.10 <= ratio <= 0.20


def test_retired_feature_flags_are_no_longer_settings_fields() -> None:
    """ADR 036 retired both flags; the features are unconditional."""
    assert "embedding_cache_enabled" not in Settings.model_fields
    assert "context_compression_enabled" not in Settings.model_fields


def test_retired_yaml_flags_warn_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing configs must keep loading: a retired key warns and is ignored."""
    # Leave the repo root so its .env does not outrank the YAML under test.
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "embedding:\n  cache_enabled: true\nretrieval:\n"
        "  context_compression_enabled: true\n  rrf_k: 77\n",
        encoding="utf-8",
    )
    with pytest.warns(DeprecationWarning, match="retired"):
        settings = load_settings(config)
    # The surviving sibling key still applies, proving only the retired key was skipped.
    assert settings.rrf_k == 77


def test_retired_cli_override_warns_instead_of_failing(tmp_path: Path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("{}\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="retired"):
        settings = load_settings(config, {"context_compression_enabled": True})
    assert settings.rag_top_k == Settings().rag_top_k


def test_every_grouped_leaf_has_exactly_one_flat_name() -> None:
    """The flat map must be total: no grouped field may be unreachable by env var."""
    leaves: set[str] = set()

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            annotation = field.annotation
            path = f"{prefix}{name}"
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                walk(annotation, f"{path}.")
            else:
                leaves.add(path)

    for group, field in Settings.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            walk(annotation, f"{group}.")

    assert leaves == set(PATH_TO_FLAT), (
        f"unmapped grouped fields: {sorted(leaves - set(PATH_TO_FLAT))}; "
        f"stale map entries: {sorted(set(PATH_TO_FLAT) - leaves)}"
    )
    assert set(Settings.model_fields) - set(FLAT_TO_PATH) >= UNGROUPED_FIELDS


@pytest.mark.parametrize(
    ("variable", "value", "attribute", "expected"),
    [
        ("HYDE_ENABLED", "true", "hyde_enabled", True),
        ("RAG_TOP_K", "11", "rag_top_k", 11),
        ("ADAPTIVE_MAX_ROUNDS", "2", "adaptive_max_rounds", 2),
        ("OTEL_SERVICE_NAME", "svc", "otel_service_name", "svc"),
        ("LLM_BACKEND", "openai", "llm_backend", "openai"),
    ],
)
def test_documented_flat_env_vars_still_resolve(
    variable: str,
    value: str,
    attribute: str,
    expected: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouping the model must not break any documented environment variable."""
    monkeypatch.setenv(variable, value)
    assert getattr(Settings(), attribute) == expected


def test_with_overrides_reaches_grouped_fields() -> None:
    """model_copy on a flat name would silently no-op; with_overrides must not."""
    settings = Settings().with_overrides(hyde_enabled=True, rag_top_k=8)
    assert settings.hyde_enabled is True
    assert settings.retrieval.hyde.enabled is True
    assert settings.retrieval.top_k == 8


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
    # The suite disables the .env layer by default (see the _isolate_configuration
    # fixture); this test is about that layer, so it opts back in. The chdir above
    # makes tmp_path's .env the only one reachable.
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")

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

    assert snapshot["api"]["key"] == "<redacted>"
    assert snapshot["llm"]["openai_api_key"] == "<redacted>"
    assert str(tmp_path) not in json.dumps(snapshot)
    assert snapshot["chroma"]["persist_path"] == "<path>"
    # Redaction must survive the flat projection too.
    flat = settings.flat_snapshot()
    assert flat["api_key"] == "<redacted>"
    assert flat["chroma_persist_path"] == "<path>"


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


def test_fallback_backend_must_be_supported_and_distinct() -> None:
    with pytest.raises(ValueError, match="must differ"):
        Settings(llm_backend="ollama", llm_fallback_backend="ollama")

    with pytest.raises(ValueError, match="llm_fallback_backend"):
        Settings(llm_fallback_backend="not-a-provider")
