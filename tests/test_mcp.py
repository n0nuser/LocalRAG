from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError

from localrag.application.repository import ChromaCollectionRepository
from localrag.ingestion.service import IngestionService
from localrag.mcp.server import build_mcp_server
from localrag.settings import Settings
from localrag.storage.vector_store import VectorStore


@dataclass
class StubStore:
    names: list[str]

    def list_collections(self) -> list[str]:
        return self.names

    def delete_collection(self, _name: str) -> None:
        return None


def make_settings(tmp_path: Path, api_key: str = "secret") -> Settings:
    return Settings(api_key=api_key, ingest_roots=[str(tmp_path / "allowed")])


def make_client(tmp_path: Path, api_key: str = "secret") -> Client:
    settings = make_settings(tmp_path, api_key=api_key)
    repo = ChromaCollectionRepository(cast("VectorStore", StubStore(["localrag"])))
    mcp = build_mcp_server(settings, collection_repo_factory=lambda: repo)
    return Client(mcp)


async def test_mcp_tool_listing_returns_the_four_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_API_KEY", "secret")
    async with make_client(tmp_path) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "search_documents",
        "answer_question",
        "ingest_path",
        "list_collections",
    }


async def test_mcp_tool_listing_requires_the_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tools/list`` is gated, not just ``tools/call``.

    The hand-rolled adapter this replaced rejected unauthenticated callers on
    every method, and the tool list discloses deployment surface, so an
    unauthenticated caller must not be able to enumerate it.
    """
    monkeypatch.delenv("MCP_API_KEY", raising=False)

    # Gating at ``on_request`` covers ``initialize`` too, so an unauthenticated
    # caller is rejected while opening the session - it never reaches a request.
    with pytest.raises(McpError, match=re.escape("Invalid request parameters")):
        async with make_client(tmp_path) as client:
            await client.list_tools()


async def test_mcp_list_collections_returns_stubbed_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_API_KEY", "secret")
    async with make_client(tmp_path) as client:
        result = await client.call_tool("list_collections", {})

    assert result.data == {"collections": ["localrag"]}


async def test_mcp_ingest_path_outside_ingest_roots_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the real application-layer path policy, not a mock of it.

    The ingestion-service stub is never actually used: ``application.service``
    raises ``IngestError`` before calling any of its methods, so a bare
    ``object()`` reference is enough to prove the rejection is real.
    """
    monkeypatch.setenv("MCP_API_KEY", "secret")
    outside = tmp_path / "outside"
    outside.mkdir()

    settings = make_settings(tmp_path)
    repo = ChromaCollectionRepository(cast("VectorStore", StubStore(["localrag"])))
    ingestion_service = cast("IngestionService", object())
    mcp = build_mcp_server(
        settings,
        ingestion_service_factory=lambda: ingestion_service,
        collection_repo_factory=lambda: repo,
    )

    async with Client(mcp) as client:
        with pytest.raises(
            ToolError, match=re.escape("Path is not under configured ingest roots.")
        ):
            await client.call_tool("ingest_path", {"path": str(outside)})


@pytest.mark.parametrize(
    ("env_key", "expect_error"),
    [
        (None, True),
        ("wrong-key", True),
        ("secret", False),
    ],
)
async def test_mcp_api_key_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_key: str | None,
    expect_error: bool,
) -> None:
    monkeypatch.delenv("MCP_API_KEY", raising=False)
    if env_key is not None:
        monkeypatch.setenv("MCP_API_KEY", env_key)

    if expect_error:
        # Gating happens at ``on_request``, which covers ``initialize``: the
        # session itself fails to open, so the rejection surfaces as a
        # protocol-level ``McpError`` rather than a tool-level error result.
        with pytest.raises(McpError, match=re.escape("Invalid request parameters")):
            async with make_client(tmp_path) as client:
                await client.call_tool("list_collections", {})
    else:
        async with make_client(tmp_path) as client:
            result = await client.call_tool("list_collections", {})
            assert result.data == {"collections": ["localrag"]}


async def test_mcp_no_api_key_configured_allows_any_caller(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, api_key="")
    repo = ChromaCollectionRepository(cast("VectorStore", StubStore(["localrag"])))
    mcp = build_mcp_server(settings, collection_repo_factory=lambda: repo)

    async with Client(mcp) as client:
        result = await client.call_tool("list_collections", {})

    assert result.data == {"collections": ["localrag"]}
