from __future__ import annotations

import os

from localrag.application.container import (
    get_collection_repository,
    get_engine,
    get_ingestion_service,
)
from localrag.mcp.server import build_mcp_server
from localrag.settings import load_settings, set_current_settings


def main() -> None:
    settings = load_settings(os.environ.get("LOCALRAG_CONFIG"))
    set_current_settings(settings)
    mcp = build_mcp_server(
        settings=settings,
        engine_factory=get_engine,
        ingestion_service_factory=get_ingestion_service,
        collection_repo_factory=get_collection_repository,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
