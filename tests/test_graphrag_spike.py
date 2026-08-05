from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.graphrag_spike import graphrag


def chunks() -> list[graphrag.SourceChunk]:
    return [
        graphrag.SourceChunk("doc-a", "chunk-1", "Ada works at Acme.", "cite-1"),
        graphrag.SourceChunk("doc-b", "chunk-2", "ACME employs Ada.", "cite-2"),
        graphrag.SourceChunk("doc-c", "chunk-3", "Disconnected fact.", "cite-3"),
    ]


def extraction(chunk_id: str, *, duplicate: bool = False) -> graphrag.Extraction:
    entities = [
        {"name": "Ada", "entity_type": "Person", "confidence": 0.9, "citation_ids": []},
        {"name": "Acme", "entity_type": "Org", "confidence": 0.8, "citation_ids": []},
    ]
    if duplicate:
        entities.append(
            {"name": " acme ", "entity_type": "org", "confidence": 0.7, "citation_ids": []}
        )
    return graphrag.Extraction.model_validate(
        {
            "chunk_id": chunk_id,
            "entities": entities,
            "relations": [
                {
                    "source": "Ada",
                    "target": "Acme",
                    "predicate": "works_at",
                    "confidence": 0.8,
                    "citation_ids": [],
                }
            ],
        }
    )


def test_entities_and_directed_relations_deduplicate_with_provenance() -> None:
    artifact = graphrag.build_graph(
        chunks()[:2],
        [extraction("chunk-1", duplicate=True), extraction("chunk-2")],
        graphrag.GraphConfig(),
    )
    assert len(artifact.nodes) == 2
    assert len(artifact.edges) == 1
    assert artifact.nodes[
        next(key for key in artifact.nodes if artifact.nodes[key].normalized_name == "acme")
    ].provenance.chunk_ids == ("chunk-1", "chunk-2")
    assert (
        artifact.edges[next(iter(artifact.edges))].source_id
        != artifact.edges[next(iter(artifact.edges))].target_id
    )


def test_empty_and_disconnected_graphs_are_safe() -> None:
    artifact = graphrag.build_graph(chunks(), [], graphrag.GraphConfig())
    assert artifact.nodes == {}
    assert graphrag.retrieve_graph(artifact, {"missing"}) == []


def test_malformed_provider_output_is_quarantined_and_retried() -> None:
    calls = 0

    def provider(items: list[graphrag.SourceChunk]) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {items[0].chunk_id: "not json"}

    accepted, failed = graphrag.ExtractionRunner(provider).run(chunks()[:1], graphrag.GraphConfig())
    assert accepted == []
    assert failed == {"chunk-1": "malformed_output: ValidationError"}
    assert calls == 2


def test_missing_files_and_oversized_inputs_fail_safely(tmp_path: Path) -> None:
    missing = graphrag.SourceChunk("doc", "chunk", "x", "cite", str(tmp_path / "gone"))  # type: ignore[attr-defined]
    with pytest.raises(FileNotFoundError):
        graphrag.build_graph([missing], [], graphrag.GraphConfig())
    oversized = graphrag.SourceChunk("doc", "big", "12345", "cite")
    accepted, failed = graphrag.ExtractionRunner(lambda _items: {}).run(
        [oversized], graphrag.GraphConfig(max_chars_per_chunk=4)
    )
    assert accepted == []
    assert failed["big"] == "input_too_large"


def test_atomic_round_trip_and_stale_schema_rejection(tmp_path: Path) -> None:
    artifact = graphrag.build_graph(chunks(), [extraction("chunk-1")], graphrag.GraphConfig())
    path = tmp_path / "graph.json"
    artifact.save_atomic(path)
    original = path.read_text()
    with pytest.raises(OSError, match="simulated interrupted write"):
        artifact.save_atomic(path, fail_before_replace=True)
    assert path.read_text() == original
    assert graphrag.GraphArtifact.load(path).artifact_id == artifact.artifact_id
    payload = json.loads(original)
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported graph schema"):
        graphrag.GraphArtifact.load(path)


def test_updates_deletes_and_compatibility_are_explicit() -> None:
    config = graphrag.GraphConfig()
    artifact = graphrag.build_graph(chunks(), [extraction("chunk-1")], config)
    assert artifact.is_compatible(artifact.corpus_id, artifact.extractor_identity, config)
    assert not artifact.is_compatible(artifact.corpus_id, "ollama:other", config)
    assert artifact.without_sources({"doc-a"}).nodes == {}


def test_bounded_filtered_traversal_and_context_citations() -> None:
    artifact = graphrag.build_graph(
        chunks(), [extraction("chunk-1")], graphrag.GraphConfig(max_hops=1)
    )
    ada = graphrag.retrieve_graph(artifact, {"ADA"}, metadata={"doc-a"})
    assert {node.normalized_name for node in ada} == {"ada", "acme"}
    with pytest.raises(ValueError, match="max_hops exceeds configured bound"):
        graphrag.retrieve_graph(artifact, {"ada"}, max_hops=2)
    hits = graphrag.compose_retrieval(
        ada, [], {chunk.chunk_id: chunk for chunk in chunks()}, context_max_chars=100
    )
    assert [hit.citation_id for hit in hits] == ["cite-1"]


def test_fixture_build_is_deterministic_and_classic_fallback_is_preserved() -> None:
    first = graphrag.build_graph(chunks(), [extraction("chunk-1")], graphrag.GraphConfig())
    second = graphrag.build_graph(
        list(reversed(chunks())), [extraction("chunk-1")], graphrag.GraphConfig()
    )
    assert first.artifact_id == second.artifact_id
    classic = graphrag.RetrievalHit("classic", "doc-a", "cite-1", 1.0)
    result = graphrag.compose_retrieval([], [classic], {}, context_max_chars=100)
    assert result == [classic]
