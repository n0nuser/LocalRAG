"""Dependency-free GraphRAG feasibility prototype.

This module is intentionally outside ``localrag``. It defines inspectable
contracts for a possible graph index without changing ingestion, Chroma, or
the default retriever.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


def normalize(value: str) -> str:
    """Normalize identity text without claiming aliases are equivalent."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ExtractionEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    citation_ids: list[str] = Field(default_factory=list)


class ExtractionRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    citation_ids: list[str] = Field(default_factory=list)


class Extraction(BaseModel):
    """The only accepted provider response shape."""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)
    entities: list[ExtractionEntity] = Field(default_factory=list)
    relations: list[ExtractionRelation] = Field(default_factory=list)


@dataclass(frozen=True)
class SourceChunk:
    source_id: str
    chunk_id: str
    text: str
    citation_id: str
    source_path: str | None = None


@dataclass(frozen=True)
class Provenance:
    source_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]
    extractor: str
    prompt_version: str
    confidence: float


@dataclass(frozen=True)
class Node:
    node_id: str
    name: str
    normalized_name: str
    entity_type: str
    provenance: Provenance


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source_id: str
    target_id: str
    predicate: str
    normalized_predicate: str
    provenance: Provenance


@dataclass(frozen=True)
class GraphConfig:
    max_chars_per_chunk: int = 4000
    max_neighbors: int = 20
    max_hops: int = 2
    context_max_chars: int = 2400
    prompt_version: str = "graph-extraction-v1"


@dataclass
class GraphArtifact:
    schema_version: int
    artifact_id: str
    corpus_id: str
    extractor_identity: str
    config: GraphConfig
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    quarantined: dict[str, str] = field(default_factory=dict)
    build_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save_atomic(self, path: Path, *, fail_before_replace: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if fail_before_replace:
                raise OSError("simulated interrupted write")
            Path(temporary).replace(path)
        finally:
            temporary_path = Path(temporary)
            if temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: Path) -> GraphArtifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            message = f"unsupported graph schema: {payload.get('schema_version')!r}"
            raise ValueError(message)
        return cls(
            schema_version=payload["schema_version"],
            artifact_id=payload["artifact_id"],
            corpus_id=payload["corpus_id"],
            extractor_identity=payload["extractor_identity"],
            config=GraphConfig(**payload["config"]),
            nodes={key: _node(value) for key, value in payload["nodes"].items()},
            edges={key: _edge(value) for key, value in payload["edges"].items()},
            quarantined=payload.get("quarantined", {}),
            build_stats=payload.get("build_stats", {}),
        )

    def without_sources(self, source_ids: set[str]) -> GraphArtifact:
        nodes = {
            key: node
            for key, node in self.nodes.items()
            if not set(node.provenance.source_ids) & source_ids
        }
        edges = {
            key: edge
            for key, edge in self.edges.items()
            if edge.source_id in nodes and edge.target_id in nodes
        }
        return _artifact(
            self.corpus_id, self.extractor_identity, self.config, nodes, edges, self.quarantined
        )

    def is_compatible(self, corpus_id: str, extractor_identity: str, config: GraphConfig) -> bool:
        return (self.corpus_id, self.extractor_identity, self.config) == (
            corpus_id,
            extractor_identity,
            config,
        )


class Provider(Protocol):
    def __call__(self, chunks: Sequence[SourceChunk]) -> dict[str, str | dict[str, Any]]: ...


class ExtractionRunner:
    """Parse strict provider output; malformed chunks are quarantined, never invented."""

    def __init__(self, provider: Provider, *, max_attempts: int = 2, batch_size: int = 8) -> None:
        self.provider, self.max_attempts, self.batch_size = provider, max_attempts, batch_size

    def run(  # noqa: C901
        self, chunks: Sequence[SourceChunk], config: GraphConfig
    ) -> tuple[list[Extraction], dict[str, str]]:
        accepted: list[Extraction] = []
        failed: dict[str, str] = {}
        for start in range(0, len(chunks), self.batch_size):
            batch = [
                chunk
                for chunk in chunks[start : start + self.batch_size]
                if len(chunk.text) <= config.max_chars_per_chunk
            ]
            for chunk in chunks[start : start + self.batch_size]:
                if len(chunk.text) > config.max_chars_per_chunk:
                    failed[chunk.chunk_id] = "input_too_large"
            for _attempt in range(self.max_attempts):
                if not batch:
                    break
                try:
                    responses = self.provider(batch)
                    parsed = {
                        chunk.chunk_id: _parse_response(responses[chunk.chunk_id])
                        for chunk in batch
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    parsed = {}
                    error = f"malformed_output: {type(exc).__name__}"
                else:
                    error = ""
                if len(parsed) == len(batch) and all(
                    parsed[chunk.chunk_id].chunk_id == chunk.chunk_id for chunk in batch
                ):
                    accepted.extend(parsed.values())
                    batch = []
                    break
                if _attempt == self.max_attempts - 1:
                    for chunk in batch:
                        failed[chunk.chunk_id] = error or "chunk_id_mismatch"
        return accepted, failed


def _prov(
    chunks: Iterable[SourceChunk], extractor: str, prompt: str, confidence: float
) -> Provenance:
    items = list(chunks)
    return Provenance(
        tuple(sorted({x.source_id for x in items})),
        tuple(sorted({x.chunk_id for x in items})),
        tuple(sorted({x.citation_id for x in items})),
        extractor,
        prompt,
        confidence,
    )


def _artifact(
    corpus_id: str,
    extractor: str,
    config: GraphConfig,
    nodes: dict[str, Node],
    edges: dict[str, Edge],
    quarantined: dict[str, str],
) -> GraphArtifact:
    identity = digest(
        {
            "corpus": corpus_id,
            "extractor": extractor,
            "config": asdict(config),
            "nodes": {key: asdict(value) for key, value in nodes.items()},
            "edges": {key: asdict(value) for key, value in edges.items()},
        }
    )
    return GraphArtifact(
        SCHEMA_VERSION,
        identity,
        corpus_id,
        extractor,
        config,
        nodes,
        edges,
        quarantined,
        {"nodes": len(nodes), "edges": len(edges), "quarantined": len(quarantined)},
    )


def build_graph(  # noqa: C901
    chunks: Sequence[SourceChunk],
    extractions: Sequence[Extraction],
    config: GraphConfig,
    *,
    extractor_identity: str = "ollama:fixture-v1",
    quarantined: dict[str, str] | None = None,
) -> GraphArtifact:
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    missing = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.source_path and not Path(chunk.source_path).is_file()
    ]
    if missing:
        message = f"missing source files for chunks: {', '.join(sorted(missing))}"
        raise FileNotFoundError(message)
    nodes: dict[str, Node] = {}
    names: dict[tuple[str, str], str] = {}
    for extraction in extractions:
        chunk = by_chunk.get(extraction.chunk_id)
        if chunk is None:
            continue
        for entity in extraction.entities:
            key = (normalize(entity.entity_type), normalize(entity.name))
            node_id = names.setdefault(key, "entity-" + digest(key)[:20])
            old = nodes.get(node_id)
            provenance = _prov(
                [chunk], extractor_identity, config.prompt_version, entity.confidence
            )
            if old:
                provenance = _merge_provenance(old.provenance, provenance)
            nodes[node_id] = Node(
                node_id, old.name if old else entity.name, key[1], key[0], provenance
            )
    edges: dict[str, Edge] = {}
    for extraction in extractions:
        chunk = by_chunk.get(extraction.chunk_id)
        if chunk is None:
            continue
        for relation in extraction.relations:
            source = names.get((normalize("entity"), normalize(relation.source))) or next(
                (key for (kind, name), key in names.items() if name == normalize(relation.source)),
                None,
            )
            target = names.get((normalize("entity"), normalize(relation.target))) or next(
                (key for (kind, name), key in names.items() if name == normalize(relation.target)),
                None,
            )
            if source is None or target is None:
                continue
            predicate = normalize(relation.predicate)
            edge_id = "edge-" + digest((source, predicate, target))[:20]
            provenance = _prov(
                [chunk], extractor_identity, config.prompt_version, relation.confidence
            )
            if edge_id in edges:
                provenance = _merge_provenance(edges[edge_id].provenance, provenance)
            edges[edge_id] = Edge(
                edge_id, source, target, relation.predicate, predicate, provenance
            )
    corpus = sorted((x.source_id, x.chunk_id, x.text) for x in chunks)
    return _artifact(digest(corpus), extractor_identity, config, nodes, edges, quarantined or {})


def _merge_provenance(first: Provenance, second: Provenance) -> Provenance:
    return Provenance(
        tuple(sorted(set(first.source_ids) | set(second.source_ids))),
        tuple(sorted(set(first.chunk_ids) | set(second.chunk_ids))),
        tuple(sorted(set(first.citation_ids) | set(second.citation_ids))),
        first.extractor,
        first.prompt_version,
        max(first.confidence, second.confidence),
    )


def _parse_response(raw: str | dict[str, Any]) -> Extraction:
    if isinstance(raw, str):
        return Extraction.model_validate_json(raw)
    return Extraction.model_validate(raw)


def retrieve_graph(
    artifact: GraphArtifact,
    query_entities: set[str],
    *,
    metadata: set[str] | None = None,
    max_hops: int | None = None,
) -> list[Node]:
    wanted = {normalize(item) for item in query_entities}
    selected = {
        node.node_id
        for node in artifact.nodes.values()
        if node.normalized_name in wanted
        and (metadata is None or metadata & set(node.provenance.source_ids))
    }
    result: list[Node] = []
    queue = deque((node_id, 0) for node_id in selected)
    seen = set(selected)
    adjacency: dict[str, list[str]] = {}
    for edge in artifact.edges.values():
        adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        adjacency.setdefault(edge.target_id, []).append(edge.source_id)
    limit = artifact.config.max_hops if max_hops is None else max_hops
    if limit < 0 or limit > artifact.config.max_hops:
        raise ValueError("max_hops exceeds configured bound")
    while queue:
        node_id, depth = queue.popleft()
        result.append(artifact.nodes[node_id])
        if depth == limit:
            continue
        for neighbor in sorted(adjacency.get(node_id, []))[: artifact.config.max_neighbors]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    return result


@dataclass(frozen=True)
class RetrievalHit:
    text: str
    source_id: str
    citation_id: str
    score: float


def compose_retrieval(
    graph_hits: Sequence[Node],
    classic_hits: Sequence[RetrievalHit],
    chunks: dict[str, SourceChunk],
    *,
    graph_weight: float = 0.25,
    context_max_chars: int = 2400,
) -> list[RetrievalHit]:
    """Opt-in composition: graph evidence adds citations, classic retrieval remains fallback."""
    if not 0 <= graph_weight <= 1:
        raise ValueError("graph_weight must be between 0 and 1")
    candidates = list(classic_hits)
    known = {hit.citation_id for hit in candidates}
    for node in graph_hits:
        for chunk_id in node.provenance.chunk_ids:
            chunk = chunks[chunk_id]
            if chunk.citation_id not in known:
                candidates.append(
                    RetrievalHit(chunk.text, chunk.source_id, chunk.citation_id, graph_weight)
                )
                known.add(chunk.citation_id)
    result: list[RetrievalHit] = []
    used = 0
    for hit in sorted(candidates, key=lambda item: (-item.score, item.citation_id)):
        if used + len(hit.text) > context_max_chars:
            continue
        result.append(hit)
        used += len(hit.text)
    return result


def _node(value: dict[str, Any]) -> Node:
    return Node(
        value["node_id"],
        value["name"],
        value["normalized_name"],
        value["entity_type"],
        Provenance(**value["provenance"]),
    )


def _edge(value: dict[str, Any]) -> Edge:
    return Edge(
        value["edge_id"],
        value["source_id"],
        value["target_id"],
        value["predicate"],
        value["normalized_predicate"],
        Provenance(**value["provenance"]),
    )
