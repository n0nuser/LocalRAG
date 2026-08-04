from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localrag.api.main import app, lifespan
from localrag.cli.app import app as cli_app
from localrag.settings import clear_current_settings, get_settings

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
