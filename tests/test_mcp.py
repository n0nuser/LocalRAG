from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from localrag.application.repository import ChromaCollectionRepository
from localrag.ingestion.service import IngestionService
from localrag.mcp.server import McpServer
from localrag.rag.engine import RAGEngine
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore


@dataclass
class StubStore:
    names: list[str]

    def list_collections(self) -> list[str]:
        return self.names

    def delete_collection(self, _name: str) -> None:
        return None


def make_server(tmp_path: Path, api_key: str = "secret") -> McpServer:
    settings = Settings(api_key=api_key, ingest_roots=[str(tmp_path / "allowed")])
    return McpServer(
        settings=settings,
        engine=cast("RAGEngine", object()),
        ingestion_service=cast("IngestionService", object()),
        collection_repo=ChromaCollectionRepository(cast("VectorStore", StubStore(["localrag"]))),
    )


def test_mcp_initialize_and_tool_listing(tmp_path: Path) -> None:
    server = make_server(tmp_path)

    initialize = server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"}, "secret"
    )
    tools = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, "secret")

    assert initialize is not None
    assert initialize["result"]["capabilities"] == {"tools": {}}
    assert tools is not None
    assert [tool["name"] for tool in tools["result"]["tools"]] == [
        "search_documents",
        "answer_question",
        "ingest_path",
        "list_collections",
    ]


def test_mcp_auth_and_collection_tool(tmp_path: Path) -> None:
    server = make_server(tmp_path)

    unauthorized = server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_collections"}}
    )
    authorized = server.handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_collections"}},
        api_key="secret",
    )

    assert unauthorized is not None
    assert unauthorized["error"]["code"] == -32001
    assert authorized is not None
    assert '"collections": ["localrag"]' in authorized["result"]["content"][0]["text"]


def test_mcp_ingest_path_preserves_ingest_root_policy(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "ingest_path", "arguments": {"path": str(outside)}},
        },
        api_key="secret",
    )

    assert response is not None
    assert response["result"] == {
        "isError": True,
        "content": [{"type": "text", "text": "Path is not under configured ingest roots."}],
    }
