"""Versioned retriever plugin contract and entry-point registry.

Plugins are trusted, installed Python code and run in-process. This module does
not sandbox them or fetch packages. Install and pin plugin distributions
explicitly, then select one with ``retriever_plugin`` in LocalRAG settings.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as metadata_entry_points
from typing import Any, Protocol, TypedDict, cast

from localrag.rag.retriever import Retriever
from localrag.settings import Settings

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0"
ENTRY_POINT_GROUP = "localrag.retrievers"


def _message(template: str, *values: object) -> str:
    return template.format(*values)


class RetrievalContext(TypedDict, total=False):
    """Stable context returned by a retriever plugin."""

    text: str
    source: str
    chunk_index: int
    score: float
    distance: float
    ingested_at: str | None
    metadata: dict[str, Any]
    freshness_factor: float


class RetrieverPlugin(Protocol):
    plugin_id: str
    contract_version: str
    compatible_contract_versions: tuple[str, ...]

    def create(self, settings: Settings) -> RetrieverContract: ...


class RetrieverContract(Protocol):
    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalContext]: ...

    def close(self) -> None: ...


class PluginRegistryError(ValueError):
    """Raised when retriever plugin metadata, selection, or loading is invalid."""


class DuplicatePluginError(PluginRegistryError):
    """Raised when two installed plugins claim the same ID."""


class PluginExecutionError(PluginRegistryError):
    """Raised when a selected plugin cannot be constructed or queried."""


class _BuiltinPlugin:
    plugin_id = "builtin"
    contract_version = CONTRACT_VERSION
    compatible_contract_versions: tuple[str, ...] = (CONTRACT_VERSION,)

    @staticmethod
    def create(settings: Settings) -> RetrieverContract:
        from localrag.api.dependencies import (  # noqa: PLC0415
            get_bm25_index,
            get_embedder,
            get_reranker,
            get_vector_store,
        )

        return cast(
            "RetrieverContract",
            Retriever(
                settings=settings,
                embedder=get_embedder(),
                vector_store=get_vector_store(),
                bm25_index=get_bm25_index(),
                reranker=get_reranker(),
            ),
        )


def _validate_plugin(plugin: object, entry_name: str) -> RetrieverPlugin:
    required = ("plugin_id", "contract_version", "compatible_contract_versions", "create")
    if any(not hasattr(plugin, attribute) for attribute in required):
        raise PluginRegistryError(
            _message("Retriever plugin '{}' has malformed metadata", entry_name)
        )
    candidate = cast("RetrieverPlugin", plugin)
    if candidate.plugin_id != entry_name:
        raise PluginRegistryError(
            _message(
                "Retriever plugin entry point '{}' metadata ID is '{}'",
                entry_name,
                candidate.plugin_id,
            )
        )
    if candidate.contract_version != CONTRACT_VERSION:
        raise PluginRegistryError(
            _message(
                "Retriever plugin '{}' has incompatible contract version '{}'",
                entry_name,
                candidate.contract_version,
            )
        )
    if CONTRACT_VERSION not in candidate.compatible_contract_versions:
        raise PluginRegistryError(
            _message("Retriever plugin '{}' has incompatible declared versions", entry_name)
        )
    return candidate


class RetrieverPluginRegistry:
    """Deterministic registry for installed retriever plugins."""

    def __init__(self, plugins: Iterable[RetrieverPlugin]) -> None:
        by_id: dict[str, RetrieverPlugin] = {}
        for plugin in plugins:
            if plugin.plugin_id in by_id:
                raise DuplicatePluginError(
                    _message("Duplicate retriever plugin ID '{}'", plugin.plugin_id)
                )
            by_id[plugin.plugin_id] = plugin
        self._plugins = dict(sorted(by_id.items()))
        self._instances: list[RetrieverContract] = []
        self._instance_ids: dict[int, str] = {}
        self._closed = False

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._plugins)

    def create(self, plugin_id: str, settings: Settings) -> RetrieverContract:
        if self._closed:
            raise PluginRegistryError("Retriever plugin registry is closed")
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            raise PluginRegistryError(_message("Unknown retriever plugin '{}'", plugin_id))
        try:
            instance = plugin.create(settings)
        except Exception as exc:
            raise PluginExecutionError(
                _message("Unable to create retriever plugin '{}': {}", plugin_id, exc)
            ) from exc
        self._instances.append(instance)
        self._instance_ids[id(instance)] = plugin_id
        return instance

    def retrieve(
        self,
        instance: RetrieverContract,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalContext]:
        plugin_id = self._instance_ids.get(id(instance), "unknown")
        if plugin_id == "builtin":
            return instance.retrieve(question, n_results, metadata_filter)
        try:
            return instance.retrieve(question, n_results, metadata_filter)
        except Exception as exc:
            raise PluginExecutionError(
                _message("Retriever plugin '{}' failed: {}", plugin_id, exc)
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for instance in self._instances:
            close = getattr(instance, "close", None)
            if close is not None:
                close()


class ManagedRetriever:
    """Own a selected plugin instance and its registry lifecycle."""

    def __init__(self, registry: RetrieverPluginRegistry, instance: RetrieverContract) -> None:
        self._registry = registry
        self._instance = instance

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalContext]:
        return self._registry.retrieve(self._instance, question, n_results, metadata_filter)

    def close(self) -> None:
        self._registry.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._instance, name)


def discover_retriever_plugins(
    *, entry_points: Iterable[EntryPoint] | None = None
) -> RetrieverPluginRegistry:
    """Discover installed plugins without network access, in deterministic order."""
    discovered: list[RetrieverPlugin] = [_BuiltinPlugin()]
    points = (
        list(entry_points)
        if entry_points is not None
        else list(metadata_entry_points(group=ENTRY_POINT_GROUP))
    )
    for point in sorted(points, key=lambda item: item.name):
        try:
            loaded = point.load()
        except Exception as exc:
            raise PluginExecutionError(
                _message("Unable to load retriever plugin '{}': {}", point.name, exc)
            ) from exc
        discovered.append(_validate_plugin(loaded, point.name))
    return RetrieverPluginRegistry(discovered)
