from __future__ import annotations

import os
from typing import Any

from localrag.application.container import Container
from localrag.mcp.server import build_mcp_server
from localrag.settings import get_settings, load_settings, set_current_settings


def _container_from_environment() -> Container:
    """Load settings and build this process's container; runs in the server lifespan."""
    settings = load_settings(os.environ.get("LOCALRAG_CONFIG"))
    set_current_settings(settings)
    return Container.build(settings)


def _build_app() -> Any:
    """Build the FastMCP ASGI app.

    The container is built in the server lifespan, not here, and closed when the app
    shuts down; see ``build_mcp_server``.
    """
    mcp = build_mcp_server(settings=get_settings(), build_container=_container_from_environment)
    return mcp.http_app(path="/mcp", stateless_http=True)


app = _build_app()
