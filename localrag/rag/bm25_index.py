from __future__ import annotations

import re
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


@dataclass
class Bm25Index:
    vector_store: VectorStore
    corpus_ids: list[str] = field(default_factory=list)
    corpus_documents: list[str] = field(default_factory=list)
    corpus_metadatas: list[dict[str, Any]] = field(default_factory=list)
    bm25: BM25Okapi | None = None

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
        bm25 = BM25Okapi(tokenized) if tokenized else None
        self.corpus_ids = corpus_ids
        self.corpus_documents = corpus_documents
        self.corpus_metadatas = corpus_metadatas
        self.bm25 = bm25

    def query(self, text: str, top_k: int) -> list[Bm25Hit]:
        if self.bm25 is None or top_k <= 0:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        query_text = text.strip().lower()
        if query_text:
            for index, document in enumerate(self.corpus_documents):
                if query_text in document.lower():
                    scores[index] = float(scores[index]) + 1.0
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )[:top_k]
        return [
            Bm25Hit(
                chunk_id=self.corpus_ids[index],
                text=self.corpus_documents[index],
                metadata=self.corpus_metadatas[index],
                score=float(scores[index]),
            )
            for index in ranked_indexes
        ]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
