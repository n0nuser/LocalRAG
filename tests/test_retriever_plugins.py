from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from localrag.plugins.retriever import (
    CONTRACT_VERSION,
    DuplicatePluginError,
    ManagedRetriever,
    PluginExecutionError,
    PluginRegistryError,
    RetrieverContract,
    discover_retriever_plugins,
)
from localrag.settings import Settings


@dataclass
class FakePlugin:
    plugin_id: str
    contract_version: str = CONTRACT_VERSION
    compatible_contract_versions: tuple[str, ...] = (CONTRACT_VERSION,)
    calls: int = 0
    closed: int = 0
    failure: bool = False

    def create(self, settings: Settings) -> FakePlugin:
        _ = settings
        self.calls += 1
        return self

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        _ = (question, n_results, metadata_filter)
        if self.failure:
            raise RuntimeError("fixture failed")
        return []

    def close(self) -> None:
        self.closed += 1


@dataclass
class ScopedRetriever(FakePlugin):
    """The built-in retriever's shape: it can open a copy for another collection."""

    collection: str = "localrag"

    def for_collection(self, collection: str) -> ScopedRetriever:
        return ScopedRetriever("builtin", collection=collection)


def unused_builtin(settings: Settings) -> RetrieverContract:
    raise AssertionError(settings)


def entry_point(plugin: FakePlugin, *, name: str | None = None) -> object:
    return SimpleNamespace(name=name or plugin.plugin_id, load=lambda: plugin)


def test_discovery_is_sorted_and_selected_plugin_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakePlugin("zeta")
    selected = FakePlugin("alpha")
    monkeypatch.setattr(
        "localrag.plugins.retriever.metadata_entry_points",
        lambda **_: [entry_point(first), entry_point(selected)],
    )

    registry = discover_retriever_plugins(builtin_factory=unused_builtin)

    assert registry.ids == ("alpha", "builtin", "zeta")
    instance = registry.create("alpha", Settings())
    assert instance is selected
    assert selected.calls == 1
    assert first.calls == 0


def test_duplicate_unknown_and_malformed_plugins_fail_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = FakePlugin("builtin")
    monkeypatch.setattr(
        "localrag.plugins.retriever.metadata_entry_points",
        lambda **_: [entry_point(duplicate)],
    )
    with pytest.raises(DuplicatePluginError, match="builtin"):
        discover_retriever_plugins(builtin_factory=unused_builtin)

    registry = discover_retriever_plugins(builtin_factory=unused_builtin, entry_points=[])
    with pytest.raises(PluginRegistryError, match="Unknown retriever plugin 'missing'"):
        registry.create("missing", Settings())

    malformed = SimpleNamespace(name="bad", load=lambda: object())
    with pytest.raises(PluginRegistryError, match="metadata"):
        discover_retriever_plugins(builtin_factory=unused_builtin, entry_points=[malformed])


@pytest.mark.parametrize(
    ("version", "compatible"),
    [("2.0", ("2.0",)), ("1.1", ("1.1",)), ("1.0", ("2.0",))],
)
def test_incompatible_versions_are_rejected(version: str, compatible: tuple[str, ...]) -> None:
    plugin = FakePlugin("bad", contract_version=version, compatible_contract_versions=compatible)
    with pytest.raises(PluginRegistryError, match="incompatible"):
        discover_retriever_plugins(
            builtin_factory=unused_builtin, entry_points=[entry_point(plugin)]
        )


def test_lifecycle_closes_once_on_success_and_failure() -> None:
    plugin = FakePlugin("fixture")
    registry = discover_retriever_plugins(
        builtin_factory=unused_builtin, entry_points=[entry_point(plugin)]
    )
    instance = registry.create("fixture", Settings())
    registry.close()
    registry.close()
    assert instance.closed == 1

    failing = FakePlugin("failing", failure=True)
    failing_registry = discover_retriever_plugins(
        builtin_factory=unused_builtin, entry_points=[entry_point(failing)]
    )
    failing_instance = failing_registry.create("failing", Settings())
    with pytest.raises(PluginExecutionError, match="failing"):
        failing_registry.retrieve(failing_instance, "question")
    failing_registry.close()
    assert failing_instance.closed == 1


def test_missing_optional_dependency_is_reported_without_importing_core_dependency() -> None:
    plugin = SimpleNamespace(
        plugin_id="optional",
        contract_version=CONTRACT_VERSION,
        compatible_contract_versions=(CONTRACT_VERSION,),
        create=lambda _settings: (_ for _ in ()).throw(ImportError("optional-lib")),
    )
    registry = discover_retriever_plugins(
        builtin_factory=unused_builtin, entry_points=[entry_point(plugin)]
    )
    with pytest.raises(PluginExecutionError, match="optional"):
        registry.create("optional", Settings())


def test_builtin_retriever_comes_from_the_injected_factory() -> None:
    built = FakePlugin("builtin")
    seen: list[Settings] = []

    def builtin(settings: Settings) -> RetrieverContract:
        seen.append(settings)
        return built

    settings = Settings(rag_top_k=3)
    registry = discover_retriever_plugins(builtin_factory=builtin, entry_points=[])

    assert (registry.create("builtin", settings), seen) == (built, [settings])


def test_only_the_builtin_retriever_opens_a_request_scoped_collection() -> None:
    plugin = ScopedRetriever("scoped")
    registry = discover_retriever_plugins(
        builtin_factory=lambda _settings: ScopedRetriever("builtin"),
        entry_points=[entry_point(plugin)],
    )
    builtin = ManagedRetriever(registry, registry.create("builtin", Settings()))
    third_party = ManagedRetriever(registry, registry.create("scoped", Settings()))

    assert builtin.for_collection("experiments").collection == "experiments"
    with pytest.raises(TypeError, match="built-in retriever"):
        third_party.for_collection("experiments")
