"""Small entry-point plugin used as the contract author's example."""

from __future__ import annotations

from typing import Any

from localrag.plugins.retriever import CONTRACT_VERSION, RetrievalContext
from localrag.settings import Settings


class ExampleRetriever:
    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalContext]:
        _ = (n_results, metadata_filter)
        return [
            {
                "text": question,
                "source": "example-plugin",
                "chunk_index": 0,
                "score": 1.0,
                "metadata": {"source": "example-plugin"},
            }
        ]

    def close(self) -> None:
        return None


class ExamplePlugin:
    plugin_id = "example"
    contract_version = CONTRACT_VERSION
    compatible_contract_versions = (CONTRACT_VERSION,)

    @staticmethod
    def create(settings: Settings) -> ExampleRetriever:
        _ = settings
        return ExampleRetriever()


plugin = ExamplePlugin()
