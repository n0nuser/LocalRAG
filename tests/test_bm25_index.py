from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest

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


def test_refresh_publishes_complete_snapshot_during_concurrent_queries() -> None:
    store = StubStore(chunks=[("old", "old corpus", {"source": "old"})])
    index = Bm25Index.from_vector_store(store)  # type: ignore[arg-type]

    def refresh() -> None:
        for _ in range(20):
            store.chunks = [("new", "new corpus", {"source": "new"})]
            index.refresh()
            store.chunks = [("old", "old corpus", {"source": "old"})]
            index.refresh()

    def query() -> set[str]:
        seen: set[str] = set()
        for _ in range(100):
            hits = index.query("corpus", top_k=1)
            if hits:
                seen.add(hits[0].chunk_id)
        return seen

    # Run the two operations concurrently without allowing a mutable corpus to
    # leak into an individual query result.
    with ThreadPoolExecutor(max_workers=2) as executor:
        query_future = executor.submit(query)
        refresh_future = executor.submit(refresh)
        refresh_future.result()
        observed = query_future.result()
    assert observed <= {"old", "new"}


def _corpus_with_outranked_minority() -> StubStore:
    # Every "public" row repeats the query term, so all of them outrank every
    # "private" row; a filter applied after top-k would see no private rows.
    public = [
        (f"public-{n}", "lantern lantern lantern lantern", {"source": "public.md"})
        for n in range(10)
    ]
    private = [
        ("private-0", "a lantern by the door and a coat", {"source": "private.md"}),
        ("private-1", "the lantern was lit at dusk", {"source": "private.md"}),
        ("private-2", "nothing relevant here", {"source": "private.md"}),
    ]
    return StubStore(chunks=public + private)


def _is_private(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("source") == "private.md"


def _never(_metadata: Mapping[str, Any]) -> bool:
    return False


@pytest.mark.parametrize(
    ("matches", "top_k", "expected_ids"),
    [
        (_is_private, 2, ["private-1", "private-0"]),
        (_is_private, 5, ["private-1", "private-0", "private-2"]),
        (_never, 5, []),
        (None, 2, ["public-0", "public-1"]),
    ],
)
def test_query_restricts_ranking_to_matching_rows_before_top_k(
    matches: Callable[[Mapping[str, Any]], bool] | None,
    top_k: int,
    expected_ids: list[str],
) -> None:
    index = Bm25Index.from_vector_store(_corpus_with_outranked_minority())  # type: ignore[arg-type]

    hits = index.query("lantern", top_k=top_k, matches=matches)

    assert [hit.chunk_id for hit in hits] == expected_ids


def test_masked_query_keeps_corpus_wide_scores_and_order() -> None:
    index = Bm25Index.from_vector_store(_corpus_with_outranked_minority())  # type: ignore[arg-type]

    unmasked = index.query("lantern", top_k=13)
    masked = index.query("lantern", top_k=13, matches=_is_private)

    assert masked == [hit for hit in unmasked if _is_private(hit.metadata)]
