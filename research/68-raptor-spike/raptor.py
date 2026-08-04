"""Dependency-free RAPTOR feasibility prototype.

This module is deliberately outside ``localrag``: it is an inspectable research
artifact, not a production retrieval or Chroma integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
SummaryFn = Callable[[str, int], str | None]
EmbedFn = Callable[[str], tuple[float, ...]]


def _raise_simulated_write_failure() -> None:
    raise OSError("simulated interrupted write")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _node_id(kind: str, level: int, children: Iterable[str], text: str = "") -> str:
    child_key = "|".join(sorted(children))
    value = f"{kind}|{level}|{text}|{child_key}"
    return f"{kind}-{hashlib.sha256(value.encode()).hexdigest()[:20]}"


@dataclass(frozen=True)
class LeafChunk:
    chunk_id: str
    source_id: str
    content: str
    content_hash: str
    embedding: tuple[float, ...]

    @classmethod
    def create(
        cls, source_id: str, chunk_id: str, content: str, embedding: tuple[float, ...]
    ) -> LeafChunk:
        return cls(
            chunk_id, source_id, content, hashlib.sha256(content.encode()).hexdigest(), embedding
        )


@dataclass(frozen=True)
class RaptorNode:
    node_id: str
    kind: str
    level: int
    text: str
    embedding: tuple[float, ...]
    child_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    summary_status: str = "not_applicable"


@dataclass(frozen=True)
class RaptorConfig:
    clustering: str = "deterministic_hash_partition"
    cluster_count: int = 2
    minimum_cluster_size: int = 2
    reduction_factor: int = 2
    seed: int = 42
    max_levels: int = 4
    summary_max_chars: int = 800
    context_max_chars: int = 2400
    level_weights: tuple[float, ...] = (1.0, 0.85, 0.7)
    prompt_version: str = "raptor-summary-v1"


@dataclass
class RaptorArtifact:
    schema_version: int
    artifact_id: str
    corpus_digest: str
    embedding_identity: str
    summarizer_identity: str
    config: RaptorConfig
    nodes: dict[str, RaptorNode]
    levels: dict[int, tuple[str, ...]]
    leaves: dict[str, LeafChunk]
    invalidation_fingerprint: str
    build_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "corpus_digest": self.corpus_digest,
            "embedding_identity": self.embedding_identity,
            "summarizer_identity": self.summarizer_identity,
            "config": asdict(self.config),
            "nodes": {key: asdict(value) for key, value in self.nodes.items()},
            "levels": {str(key): value for key, value in self.levels.items()},
            "leaves": {key: asdict(value) for key, value in self.leaves.items()},
            "invalidation_fingerprint": self.invalidation_fingerprint,
            "build_stats": self.build_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RaptorArtifact:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported RAPTOR artifact schema")
        config = RaptorConfig(**data["config"])  # type: ignore[arg-type]
        nodes = {key: RaptorNode(**value) for key, value in data["nodes"].items()}  # type: ignore[union-attr]
        leaves = {key: LeafChunk(**value) for key, value in data["leaves"].items()}  # type: ignore[union-attr]
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_id=data["artifact_id"],
            corpus_digest=data["corpus_digest"],
            embedding_identity=data["embedding_identity"],
            summarizer_identity=data["summarizer_identity"],
            config=config,
            nodes=nodes,
            levels={int(key): tuple(value) for key, value in data["levels"].items()},  # type: ignore[union-attr]
            leaves=leaves,
            invalidation_fingerprint=data["invalidation_fingerprint"],
            build_stats=data.get("build_stats", {}),
        )

    def save_atomic(self, path: Path, *, fail_before_replace: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if fail_before_replace:
                _raise_simulated_write_failure()
            Path(temporary).replace(path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> RaptorArtifact:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def is_compatible(
        self, *, embedding_identity: str, summarizer_identity: str, config: RaptorConfig
    ) -> bool:
        return self.invalidation_fingerprint == _fingerprint(
            self.leaves.values(), embedding_identity, summarizer_identity, config
        )

    def without_sources(self, source_ids: set[str]) -> RaptorArtifact:
        kept = [leaf for leaf in self.leaves.values() if leaf.source_id not in source_ids]
        return RaptorBuilder(self.config, self.embedding_identity, self.summarizer_identity).build(
            kept
        )

    def replace_sources(self, replacements: Iterable[LeafChunk]) -> RaptorArtifact:
        replacement_list = list(replacements)
        replaced_ids = {leaf.source_id for leaf in replacement_list}
        kept = [leaf for leaf in self.leaves.values() if leaf.source_id not in replaced_ids]
        return RaptorBuilder(self.config, self.embedding_identity, self.summarizer_identity).build(
            [*kept, *replacement_list]
        )


class RaptorBuilder:
    def __init__(
        self,
        config: RaptorConfig,
        embedding_identity: str,
        summarizer_identity: str,
        summary: SummaryFn | None = None,
    ) -> None:
        if config.minimum_cluster_size < 2 or config.reduction_factor < 2 or config.max_levels < 1:
            raise ValueError("cluster size, reduction factor, and max levels must be valid")
        self.config = config
        self.embedding_identity = embedding_identity
        self.summarizer_identity = summarizer_identity
        self.summary = summary or (lambda text, _level: text[: config.summary_max_chars])

    def build(self, leaves: Iterable[LeafChunk]) -> RaptorArtifact:
        current = sorted(leaves, key=lambda leaf: leaf.chunk_id)
        leaf_map = {leaf.chunk_id: leaf for leaf in current}
        nodes = {leaf.chunk_id: self._leaf_node(leaf) for leaf in current}
        levels: dict[int, tuple[str, ...]] = {0: tuple(leaf.chunk_id for leaf in current)}
        calls = 0
        failed = 0
        for level in range(1, self.config.max_levels + 1):
            if len(current) < self.config.minimum_cluster_size * 2:
                break
            clusters = self._clusters(current, level)
            next_nodes: list[RaptorNode] = []
            for cluster in clusters:
                if len(cluster) < self.config.minimum_cluster_size:
                    continue
                children = tuple(item.chunk_id for item in cluster)
                text = "\n".join(nodes[item.chunk_id].text for item in cluster)
                calls += 1
                summary = self.summary(text, level)
                if not summary:
                    failed += 1
                    continue
                summary = summary[: self.config.summary_max_chars]
                source_chunks = tuple(
                    sorted(
                        {sid for item in cluster for sid in nodes[item.chunk_id].source_chunk_ids}
                    )
                )
                source_ids = tuple(
                    sorted({sid for item in cluster for sid in nodes[item.chunk_id].source_ids})
                )
                hashes = tuple(
                    sorted(
                        {value for item in cluster for value in nodes[item.chunk_id].content_hashes}
                    )
                )
                node = RaptorNode(
                    _node_id("summary", level, children, summary),
                    "summary",
                    level,
                    summary,
                    self._mean_embedding(cluster),
                    children,
                    source_chunks,
                    source_ids,
                    hashes,
                    "complete",
                )
                nodes[node.node_id] = node
                next_nodes.append(node)
            if not next_nodes or len(next_nodes) >= len(current):
                break
            current = next_nodes
            levels[level] = tuple(node.node_id for node in current)
        fingerprint = _fingerprint(
            leaf_map.values(), self.embedding_identity, self.summarizer_identity, self.config
        )
        return RaptorArtifact(
            SCHEMA_VERSION,
            f"raptor-{fingerprint[:20]}",
            _digest(sorted(leaf_map)),
            self.embedding_identity,
            self.summarizer_identity,
            self.config,
            nodes,
            levels,
            leaf_map,
            fingerprint,
            {
                "leaf_count": len(current) if not leaf_map else len(leaf_map),
                "summary_count": len(nodes) - len(leaf_map),
                "summary_calls": calls,
                "failed_summaries": failed,
            },
        )

    def _leaf_node(self, leaf: LeafChunk) -> RaptorNode:
        return RaptorNode(
            leaf.chunk_id,
            "leaf",
            0,
            leaf.content,
            leaf.embedding,
            (),
            (leaf.chunk_id,),
            (leaf.source_id,),
            (leaf.content_hash,),
        )

    def _clusters(
        self, items: list[LeafChunk | RaptorNode], level: int
    ) -> list[list[LeafChunk | RaptorNode]]:
        count = min(self.config.cluster_count, max(1, len(items) // self.config.reduction_factor))
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{self.config.seed}:{level}:{item.chunk_id}".encode()
            ).hexdigest(),
        )
        clusters = [[] for _ in range(count)]
        for index, item in enumerate(ordered):
            clusters[index % count].append(item)
        return clusters

    @staticmethod
    def _mean_embedding(items: list[LeafChunk | RaptorNode]) -> tuple[float, ...]:
        dimension = len(items[0].embedding)
        return tuple(
            sum(item.embedding[index] for item in items) / len(items) for index in range(dimension)
        )


def _fingerprint(
    leaves: Iterable[LeafChunk], embedding: str, summarizer: str, config: RaptorConfig
) -> str:
    identity = [
        (leaf.chunk_id, leaf.source_id, leaf.content_hash, leaf.embedding)
        for leaf in sorted(leaves, key=lambda item: item.chunk_id)
    ]
    return _digest(
        {
            "leaves": identity,
            "embedding": embedding,
            "summarizer": summarizer,
            "config": asdict(config),
        }
    )


class RaptorRetriever:
    def __init__(self, artifact: RaptorArtifact, embed: EmbedFn) -> None:
        self.artifact = artifact
        self.embed = embed

    def search(
        self, query: str, top_k: int = 5, levels: tuple[int, ...] | None = None
    ) -> list[RaptorNode]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        selected = levels or tuple(sorted(self.artifact.levels))
        query_embedding = self.embed(query)
        candidates = [
            self.artifact.nodes[node_id]
            for level in selected
            for node_id in self.artifact.levels.get(level, ())
        ]
        ranked = sorted(
            candidates,
            key=lambda node: (
                -self._cosine(query_embedding, node.embedding)
                * self.artifact.config.level_weights[
                    min(node.level, len(self.artifact.config.level_weights) - 1)
                ],
                node.node_id,
            ),
        )
        result: list[RaptorNode] = []
        seen_sources: set[str] = set()
        used = 0
        for node in ranked:
            if any(source in seen_sources for source in node.source_ids):
                continue
            if used + len(node.text) > self.artifact.config.context_max_chars:
                continue
            result.append(node)
            seen_sources.update(node.source_ids)
            used += len(node.text)
            if len(result) == top_k:
                break
        return result

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimensions must match")
        denominator = math.sqrt(
            sum(value * value for value in left) * sum(value * value for value in right)
        )
        return (
            sum(a * b for a, b in zip(left, right, strict=True)) / denominator
            if denominator
            else 0.0
        )


def citation(node: RaptorNode) -> dict[str, object]:
    """Serialize citations without allowing summary nodes to hide source IDs."""
    return {
        "node_id": node.node_id,
        "kind": node.kind,
        "source_ids": list(node.source_ids),
        "source_chunk_ids": list(node.source_chunk_ids),
        "content_hashes": list(node.content_hashes),
    }
