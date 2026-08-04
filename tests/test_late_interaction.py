from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.late_interaction import LateInteractionIndex, maxsim


def test_maxsim_matches_hand_calculated_token_maxima() -> None:
    query = ((1.0, 0.0), (0.0, 1.0))
    document = ((0.8, 0.2), (0.1, 0.9), (-1.0, 0.0))

    assert maxsim(query, document) == pytest.approx(1.7)
    assert maxsim(query, document, normalize=True) == pytest.approx(0.85)


def test_maxsim_masks_padding_and_empty_queries() -> None:
    query = ((1.0, 0.0), (0.0, 1.0), (99.0, 99.0))
    document = ((0.8, 0.2), (0.1, 0.9), (99.0, 99.0))

    assert maxsim(
        query, document, query_mask=(True, True, False), document_mask=(True, True, False)
    ) == pytest.approx(1.7)
    assert maxsim((), document) == 0.0


def test_maxsim_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        maxsim(((1.0, 0.0),), ((1.0,),))


def test_index_search_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    index = LateInteractionIndex()
    index.add("b", ((0.0, 1.0),))
    index.add("a", ((1.0, 0.0),))
    query = ((1.0, 0.0),)

    assert index.search(query, top_k=2) == [("a", 1.0), ("b", 0.0)]

    path = tmp_path / "index.json"
    index.save(path)
    loaded = LateInteractionIndex.load(path)
    assert loaded.search(query, top_k=2) == index.search(query, top_k=2)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_index_rejects_duplicate_ids_and_invalid_top_k() -> None:
    index = LateInteractionIndex()
    index.add("doc", ((1.0,),))
    with pytest.raises(ValueError, match="already exists"):
        index.add("doc", ((1.0,),))
    with pytest.raises(ValueError, match="positive"):
        index.search(((1.0,),), top_k=0)
