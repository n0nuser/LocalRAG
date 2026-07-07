from __future__ import annotations

from localrag.settings import Settings


def test_default_chunk_overlap_is_within_10_to_20_percent_of_max_chars() -> None:
    settings = Settings()
    ratio = settings.chunk_overlap_chars / settings.chunk_max_chars
    assert 0.10 <= ratio <= 0.20
