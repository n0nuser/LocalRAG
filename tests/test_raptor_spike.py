from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_68_raptor import raptor


def leaves(count: int = 6) -> list[raptor.LeafChunk]:
    return [
        raptor.LeafChunk.create(
            f"doc-{index // 2}",
            f"chunk-{index}",
            f"topic {index}",
            (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0),
        )
        for index in range(count)
    ]


def builder(
    summary: raptor.SummaryFn | None = None, config: raptor.RaptorConfig | None = None
) -> raptor.RaptorBuilder:
    return raptor.RaptorBuilder(
        config or raptor.RaptorConfig(), "embed:test:v1", "summary:test:v1", summary
    )


def test_build_preserves_provenance_and_is_deterministic() -> None:
    first = builder().build(leaves())
    second = builder().build(list(reversed(leaves())))
    assert first.artifact_id == second.artifact_id
    summaries = [node for node in first.nodes.values() if node.kind == "summary"]
    assert summaries
    assert all(node.source_chunk_ids and node.source_ids for node in summaries)
    assert all(node.kind == "leaf" for node in first.nodes.values() if node.level == 0)


@pytest.mark.parametrize("items", [[], leaves(1), leaves(3)])
def test_empty_and_small_corpora_do_not_create_summary(items: list[raptor.LeafChunk]) -> None:
    artifact = builder().build(items)
    assert not [node for node in artifact.nodes.values() if node.kind == "summary"]


def test_failed_summary_keeps_leaves_and_records_stats() -> None:
    artifact = builder(lambda _text, _level: None).build(leaves())
    assert artifact.build_stats["failed_summaries"] > 0
    assert len(artifact.leaves) == 6


def test_persistence_is_round_trip_and_atomic(tmp_path: Path) -> None:
    artifact = builder().build(leaves())
    path = tmp_path / "tree.json"
    artifact.save_atomic(path)
    original = path.read_text()
    with pytest.raises(OSError, match="simulated interrupted write"):
        artifact.save_atomic(path, fail_before_replace=True)
    assert path.read_text() == original
    assert raptor.RaptorArtifact.load(path).artifact_id == artifact.artifact_id
    assert json.loads(original)["schema_version"] == 1


def test_invalidation_and_delete_are_explicit() -> None:
    artifact = builder().build(leaves())
    assert artifact.is_compatible(
        embedding_identity="embed:test:v1",
        summarizer_identity="summary:test:v1",
        config=artifact.config,
    )
    assert not artifact.is_compatible(
        embedding_identity="embed:other",
        summarizer_identity="summary:test:v1",
        config=artifact.config,
    )
    updated = artifact.without_sources({"doc-0"})
    assert all(leaf.source_id != "doc-0" for leaf in updated.leaves.values())
    replacement = raptor.LeafChunk.create("doc-0", "chunk-new", "updated", (1.0, 0.0))
    replaced = artifact.replace_sources([replacement])
    assert {leaf.chunk_id for leaf in replaced.leaves.values() if leaf.source_id == "doc-0"} == {
        "chunk-new"
    }


def test_retrieval_deduplicates_sources_and_citations() -> None:
    artifact = builder().build(leaves())
    result = raptor.RaptorRetriever(artifact, lambda _query: (1.0, 0.0)).search("topic", top_k=10)
    assert len({source for node in result for source in node.source_ids}) == len(result)
    assert all(raptor.citation(node)["source_chunk_ids"] for node in result)


def test_retrieval_bounds_context_and_rejects_invalid_top_k() -> None:
    config = raptor.RaptorConfig(context_max_chars=5)
    artifact = builder(config=config).build(leaves(2))
    with pytest.raises(ValueError, match="top_k"):
        raptor.RaptorRetriever(artifact, lambda _query: (1.0, 0.0)).search("x", 0)
    assert raptor.RaptorRetriever(artifact, lambda _query: (1.0, 0.0)).search("x") == []


def test_oversized_summary_is_truncated() -> None:
    config = raptor.RaptorConfig(summary_max_chars=4)
    artifact = builder(lambda _text, _level: "0123456789", config).build(leaves())
    assert all(len(node.text) <= 4 for node in artifact.nodes.values() if node.kind == "summary")
