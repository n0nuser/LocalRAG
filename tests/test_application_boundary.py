from __future__ import annotations

from pathlib import Path


def test_application_and_cli_do_not_depend_on_http_adapter() -> None:
    root = Path(__file__).parents[1]
    checked = [root / "localrag" / "application", root / "localrag" / "cli"]
    forbidden = ("localrag.api", "fastapi")

    for directory in checked:
        for path in directory.rglob("*.py"):
            source = path.read_text()
            assert not any(token in source for token in forbidden), path
