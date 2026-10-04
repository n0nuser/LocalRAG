from __future__ import annotations

import os

from localrag.application.container import Container
from localrag.mcp.server import build_mcp_server
from localrag.settings import load_settings, set_current_settings


def main() -> None:
    settings = load_settings(os.environ.get("LOCALRAG_CONFIG"))
    set_current_settings(settings)
    mcp = build_mcp_server(settings=settings, build_container=lambda: Container.build(settings))
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
