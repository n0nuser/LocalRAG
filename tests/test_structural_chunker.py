from __future__ import annotations

from itertools import pairwise

from localrag.ingestion.structural_chunker import chunk_document
from localrag.settings import Settings


def test_chunk_document_markdown_keeps_table_rows_together() -> None:
    markdown_text = """
# Pricing
| Plan | Price |
| --- | --- |
| Pro | 20 |
| Team | 50 |

## Notes
Billing is monthly.
""".strip()
    settings = Settings(chunk_max_chars=1200, chunk_min_chars=50)

    chunks = chunk_document(markdown_text, ".md", settings)

    assert any("| Team | 50 |" in chunk.text for chunk in chunks)
    assert any(chunk.heading_path == "Pricing" for chunk in chunks)


def test_chunk_document_markdown_keeps_fenced_code_block() -> None:
    markdown_text = """
# API
```python
def build():
    return 1
```
""".strip()
    settings = Settings(chunk_max_chars=1200, chunk_min_chars=50)

    chunks = chunk_document(markdown_text, ".md", settings)

    assert len(chunks) == 1
    assert chunks[0].text == "# API\n\n```python\ndef build():\n    return 1\n```"
    assert chunks[0].heading_path == "API"
    assert chunks[0].chunk_type == "markdown_code"


def test_chunk_document_markdown_splits_oversized_paragraph() -> None:
    oversized = "A" * 30
    markdown_text = f"# Long\n\n{oversized}"
    settings = Settings(chunk_max_chars=10, chunk_min_chars=1)

    chunks = chunk_document(markdown_text, ".md", settings)

    assert len(chunks) > 1
    assert all(chunk.heading_path == "Long" for chunk in chunks)


def test_chunk_document_oversized_paragraph_overlaps_between_chunks() -> None:
    oversized = "A" * 30
    markdown_text = f"# Long\n\n{oversized}"
    settings = Settings(chunk_max_chars=10, chunk_min_chars=1, chunk_overlap_chars=3)

    chunks = chunk_document(markdown_text, ".md", settings)

    body_chunks = [chunk for chunk in chunks if chunk.text != "# Long"]
    assert len(body_chunks) > 1
    for prev_chunk, next_chunk in pairwise(body_chunks):
        assert prev_chunk.text[-3:] == next_chunk.text[:3]


def test_chunk_document_oversized_paragraph_splits_on_sentence_boundary() -> None:
    sentence = "This is one sentence."
    long_text = " ".join([sentence] * 6)
    markdown_text = f"# Notes\n\n{long_text}"
    settings = Settings(chunk_max_chars=30, chunk_min_chars=1, chunk_overlap_chars=0)

    chunks = chunk_document(markdown_text, ".md", settings)

    body_chunks = [chunk for chunk in chunks if chunk.text != "# Notes"]
    assert len(body_chunks) > 1
    for chunk in body_chunks:
        assert chunk.text == sentence


def test_chunk_document_oversized_paragraph_early_boundary_does_not_hang() -> None:
    # Regression test: a sentence boundary very close to the start of the
    # window, followed by a long run with no further boundaries or spaces.
    # With overlap_chars large relative to max_chars, computing the next
    # start as `end - overlap_chars` (without capping to the actual chunk
    # length) can push start backward or leave it unchanged, looping forever.
    text = "X. " + ("Y" * 5000)
    markdown_text = f"# Doc\n\n{text}"
    settings = Settings(chunk_max_chars=50, chunk_min_chars=1, chunk_overlap_chars=40)

    chunks = chunk_document(markdown_text, ".md", settings)

    body_chunks = [chunk for chunk in chunks if chunk.text != "# Doc"]
    assert len(body_chunks) > 1
    # Reaching the end of input (rather than hanging) is the actual regression check.
    assert body_chunks[-1].text.rstrip().endswith("Y")


def test_chunk_document_non_markdown_packs_paragraphs() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    settings = Settings(chunk_max_chars=40, chunk_min_chars=20)

    chunks = chunk_document(text, ".txt", settings)

    assert len(chunks) == 2
    assert chunks[0].chunk_type == "text_block"
    assert chunks[0].heading_path == ""
