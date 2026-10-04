from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from localrag.storage.vector_store import VectorStore

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_:\-./]+")


@dataclass
class Bm25Hit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class _Bm25Snapshot:
    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[dict[str, Any], ...]
    index: BM25Okapi | None


@dataclass
class Bm25Index:
    vector_store: VectorStore
    corpus_ids: list[str] = field(default_factory=list)
    corpus_documents: list[str] = field(default_factory=list)
    corpus_metadatas: list[dict[str, Any]] = field(default_factory=list)
    bm25: BM25Okapi | None = None
    _snapshot: _Bm25Snapshot = field(
        default_factory=lambda: _Bm25Snapshot((), (), (), None), init=False, repr=False
    )
    _snapshot_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_vector_store(cls, store: VectorStore) -> Bm25Index:
        index = cls(vector_store=store)
        index.refresh()
        return index

    def refresh(self) -> None:
        chunks = self.vector_store.get_all_chunks()
        corpus_ids: list[str] = []
        corpus_documents: list[str] = []
        corpus_metadatas: list[dict[str, Any]] = []
        tokenized: list[list[str]] = []
        for chunk_id, document, metadata in chunks:
            corpus_ids.append(chunk_id)
            corpus_documents.append(document)
            corpus_metadatas.append(metadata)
            tokenized.append(tokenize(document))
        snapshot = _Bm25Snapshot(
            tuple(corpus_ids),
            tuple(corpus_documents),
            tuple(corpus_metadatas),
            BM25Okapi(tokenized) if tokenized else None,
        )
        with self._snapshot_lock:
            self._snapshot = snapshot
            # Keep these attributes for callers that inspect the index state.
            self.corpus_ids = list(snapshot.ids)
            self.corpus_documents = list(snapshot.documents)
            self.corpus_metadatas = list(snapshot.metadatas)
            self.bm25 = snapshot.index

    def query(
        self,
        text: str,
        top_k: int,
        matches: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> list[Bm25Hit]:
        """Rank the chunks ``matches`` accepts, all of them when it is None.

        Rows are masked before scoring and the ``top_k`` cut, so a selective filter
        still yields up to ``top_k`` hits. IDF and average document length stay
        corpus-wide: masking changes which chunks compete, never their scores.
        """
        with self._snapshot_lock:
            snapshot = self._snapshot
        if snapshot.index is None or top_k <= 0:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        candidates = [
            index
            for index, metadata in enumerate(snapshot.metadatas)
            if matches is None or matches(metadata)
        ]

        if not candidates:
            return []

        scores = snapshot.index.get_batch_scores(tokens, candidates)
        query_text = text.strip().lower()
        if query_text:
            for position, index in enumerate(candidates):
                if query_text in snapshot.documents[index].lower():
                    scores[position] += 1.0
        ranked_positions = sorted(
            range(len(candidates)),
            key=lambda position: scores[position],
            reverse=True,
        )[:top_k]
        return [
            Bm25Hit(
                chunk_id=snapshot.ids[candidates[position]],
                text=snapshot.documents[candidates[position]],
                metadata=snapshot.metadatas[candidates[position]],
                score=float(scores[position]),
            )
            for position in ranked_positions
        ]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
