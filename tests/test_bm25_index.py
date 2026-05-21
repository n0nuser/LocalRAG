from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localrag.rag.bm25_index import Bm25Index, tokenize


@dataclass
class StubStore:
    chunks: list[tuple[str, str, dict[str, Any]]]

    def get_all_chunks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self.chunks


def test_tokenize_keeps_error_codes_as_single_tokens() -> None:
    tokens = tokenize("Got ERR_QUIC_PROTOCOL_ERROR:100 in v1.2.3")

    assert tokens == ["got", "err_quic_protocol_error:100", "in", "v1.2.3"]


def test_bm25_index_returns_exact_string_match_first() -> None:
    store = StubStore(
        chunks=[
            (
                "1",
                "General troubleshooting for networking errors.",
                {"source": "a.md", "chunk_index": 0},
            ),
            (
                "2",
                "Fix ERR_QUIC_PROTOCOL_ERROR by clearing your transport cache.",
                {"source": "b.md", "chunk_index": 1},
            ),
        ]
    )
    index = Bm25Index.from_vector_store(store)  # type: ignore[arg-type]

    hits = index.query("ERR_QUIC_PROTOCOL_ERROR", top_k=2)

    assert hits[0].chunk_id == "2"
    assert hits[0].metadata == {"source": "b.md", "chunk_index": 1}
