from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localrag.api.main import app, lifespan
from localrag.cli.app import app as cli_app
from localrag.settings import clear_current_settings, get_settings, load_settings

runner = CliRunner()


def test_cli_config_and_explicit_override_match_resolved_contract(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("tenant_id: yaml-tenant\n", encoding="utf-8")

    result = runner.invoke(
        cli_app,
        ["--config", str(config), "--set", "tenant_id=cli-tenant", "config-show"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["tenant_id"] == "cli-tenant"
    clear_current_settings()


def test_cli_unknown_override_fails_with_field_name() -> None:
    result = runner.invoke(cli_app, ["--set", "not_a_setting=true", "config-show"])

    assert result.exit_code != 0
    assert "not_a_setting" in result.output


def test_list_valued_override_resolves_and_stays_a_list() -> None:
    settings = load_settings(cli_overrides={"ingest_roots": ["/a", "/b"]})

    assert settings.ingest_roots == ["/a", "/b"]
    assert isinstance(settings.ingest_roots, list)


def test_list_valued_override_is_cached_not_rejected() -> None:
    first = load_settings(cli_overrides={"ingest_roots": ["/a", "/b"]})
    second = load_settings(cli_overrides={"ingest_roots": ["/a", "/b"]})

    assert first.ingest_roots == second.ingest_roots


def test_distinct_list_overrides_do_not_share_a_cache_entry() -> None:
    first = load_settings(cli_overrides={"ingest_roots": ["/a", "/b"]})
    second = load_settings(cli_overrides={"ingest_roots": ["/b", "/a"]})

    assert first.ingest_roots == ["/a", "/b"]
    assert second.ingest_roots == ["/b", "/a"]


def test_cli_accepts_a_list_valued_set_override() -> None:
    result = runner.invoke(cli_app, ["--set", 'ingest_roots=["/tmp/inbox"]', "config-show"])

    assert result.exit_code == 0, result.output
    # config-show redacts path values, so the redacted entry is the observable
    # signal that the list override resolved instead of crashing the cache key.
    assert json.loads(result.stdout)["ingest"]["roots"] == ["<path>"]
    clear_current_settings()


@pytest.mark.asyncio
async def test_api_startup_uses_the_same_yaml_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("tenant_id: api-tenant\n", encoding="utf-8")
    monkeypatch.setenv("LOCALRAG_CONFIG", str(config))

    async with lifespan(app):
        assert get_settings().tenant_id == "api-tenant"

    clear_current_settings()
