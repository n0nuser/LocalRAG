from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_compose_defaults_are_local_only_and_secret_backed() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    for name in ("ollama", "chromadb", "localrag-api"):
        assert all(str(port).startswith("127.0.0.1:") for port in services[name]["ports"])
    assert "API_KEY=${API_KEY:?" in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "GRAFANA_ADMIN_PASSWORD:?" in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert services["localrag-api"]["read_only"] is True
    assert services["localrag-api"]["cap_drop"] == ["ALL"]
    assert services["prometheus"]["profiles"] == ["observability"]
    assert services["grafana"]["profiles"] == ["observability"]


def test_application_image_drops_root() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER localrag" in dockerfile
    assert "useradd --create-home --uid 10001 localrag" in dockerfile
