from __future__ import annotations

from itertools import pairwise

import pytest

from localrag.chunks.record import Chunk
from localrag.chunks.strategies import chunk_source
from localrag.settings import Settings

SOURCE = "/docs/guide.md"


def test_structural_markdown_keeps_table_rows_together() -> None:
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

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    assert any("| Team | 50 |" in chunk.text for chunk in chunks)
    assert any(chunk.heading_path == "Pricing" for chunk in chunks)


def test_structural_markdown_keeps_fenced_code_block() -> None:
    markdown_text = """
# API
```python
def build():
    return 1
```
""".strip()
    settings = Settings(chunk_max_chars=1200, chunk_min_chars=50)

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    assert len(chunks) == 1
    assert chunks[0].text == "# API\n\n```python\ndef build():\n    return 1\n```"
    assert chunks[0].heading_path == "API"
    assert chunks[0].chunk_type == "markdown_code"


def test_structural_markdown_splits_oversized_paragraph() -> None:
    oversized = "A" * 30
    markdown_text = f"# Long\n\n{oversized}"
    settings = Settings(chunk_max_chars=10, chunk_min_chars=1)

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    assert len(chunks) > 1
    assert all(chunk.heading_path == "Long" for chunk in chunks)


def test_structural_oversized_paragraph_overlaps_between_chunks() -> None:
    oversized = "A" * 30
    markdown_text = f"# Long\n\n{oversized}"
    settings = Settings(chunk_max_chars=10, chunk_min_chars=1, chunk_overlap_chars=3)

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    body_chunks = [chunk for chunk in chunks if chunk.text != "# Long"]
    assert len(body_chunks) > 1
    for prev_chunk, next_chunk in pairwise(body_chunks):
        assert prev_chunk.text[-3:] == next_chunk.text[:3]


def test_structural_oversized_paragraph_splits_on_sentence_boundary() -> None:
    sentence = "This is one sentence."
    long_text = " ".join([sentence] * 6)
    markdown_text = f"# Notes\n\n{long_text}"
    settings = Settings(chunk_max_chars=30, chunk_min_chars=1, chunk_overlap_chars=0)

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    body_chunks = [chunk for chunk in chunks if chunk.text != "# Notes"]
    assert len(body_chunks) > 1
    for chunk in body_chunks:
        assert chunk.text == sentence


def test_structural_oversized_paragraph_early_boundary_does_not_hang() -> None:
    # Regression test: a sentence boundary very close to the start of the
    # window, followed by a long run with no further boundaries or spaces.
    # With overlap_chars large relative to max_chars, computing the next
    # start as `end - overlap_chars` (without capping to the actual chunk
    # length) can push start backward or leave it unchanged, looping forever.
    text = "X. " + ("Y" * 5000)
    markdown_text = f"# Doc\n\n{text}"
    settings = Settings(chunk_max_chars=50, chunk_min_chars=1, chunk_overlap_chars=40)

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    body_chunks = [chunk for chunk in chunks if chunk.text != "# Doc"]
    assert len(body_chunks) > 1
    # Reaching the end of input (rather than hanging) is the actual regression check.
    assert body_chunks[-1].text.rstrip().endswith("Y")


def test_structural_non_markdown_packs_paragraphs() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    settings = Settings(chunk_max_chars=40, chunk_min_chars=20)

    chunks = chunk_source(text, ".txt", SOURCE, settings)

    assert len(chunks) == 2
    assert chunks[0].chunk_type == "text_block"
    assert chunks[0].heading_path == ""


def test_structural_anydoc_output_follows_markdown_headings() -> None:
    anydoc_markdown = "# Sheet summary\n\n| Name | Value |\n| --- | --- |\n| A | 1 |"
    settings = Settings(chunk_max_chars=1200, chunk_min_chars=1)

    chunks = chunk_source(anydoc_markdown, ".xlsx", SOURCE, settings)

    assert chunks[0].heading_path == "Sheet summary"
    assert chunks[0].chunk_type == "markdown_table"


def test_structural_pdf_follows_markdown_headings() -> None:
    """The PDF parser emits Markdown, so PDFs must chunk by heading structure.

    Without this routing the extracted headings are discarded and every PDF
    chunk carries an empty heading_path.
    """
    pdf_markdown = """
# III. ADMINISTRACION LOCAL

## C. ANUNCIOS

### AYUNTAMIENTO DE CABRERIZOS

Se somete a informacion publica el expediente.
""".strip()
    settings = Settings(chunk_max_chars=1200, chunk_min_chars=20)

    chunks = chunk_source(pdf_markdown, ".pdf", SOURCE, settings)

    assert any(
        chunk.heading_path == "III. ADMINISTRACION LOCAL > C. ANUNCIOS > AYUNTAMIENTO DE CABRERIZOS"
        for chunk in chunks
    )


def test_structural_pdf_without_headings_falls_back_to_text_blocks() -> None:
    """An OCR'd scan yields no Markdown headings and must not be mislabelled."""
    text = "\n\n".join(["Parrafo escaneado sin encabezado alguno." for _ in range(4)])
    settings = Settings(chunk_max_chars=60, chunk_min_chars=20)

    chunks = chunk_source(text, ".pdf", SOURCE, settings)

    assert chunks
    assert all(chunk.chunk_type == "text_block" for chunk in chunks)
    assert all(chunk.heading_path == "" for chunk in chunks)


def test_structural_markdown_preamble_before_first_heading_is_text_block() -> None:
    """Content ahead of the first heading is prose, not a markdown section."""
    markdown_text = "Preamble prose.\n\n# Real Heading\n\nBody under the heading."
    settings = Settings(chunk_max_chars=1200, chunk_min_chars=1)

    chunks = chunk_source(markdown_text, ".md", SOURCE, settings)

    preamble = next(chunk for chunk in chunks if "Preamble" in chunk.text)
    assert preamble.chunk_type == "text_block"
    assert preamble.heading_path == ""


def test_fixed_chunks_overlap_by_the_configured_characters() -> None:
    settings = Settings(chunking_mode="fixed", chunk_chars=10, chunk_overlap_chars=2)

    chunks = chunk_source("abcdefghijklmnopqrstuvwxyz", ".txt", SOURCE, settings)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz", "yz"]


def test_fixed_chunking_without_a_positive_size_emits_the_whole_text() -> None:
    settings = Settings(chunking_mode="fixed", chunk_chars=0, chunk_overlap_chars=10)

    chunks = chunk_source("  abc  ", ".txt", SOURCE, settings)

    assert [chunk.text for chunk in chunks] == ["abc"]


@pytest.mark.parametrize("mode", ["fixed", "structural", "recursive"])
def test_empty_input_emits_no_chunks(mode: str) -> None:
    settings = Settings(chunking_mode=mode, chunk_min_chars=1)

    assert chunk_source(" \n ", ".txt", SOURCE, settings) == []


def test_recursive_chunks_are_placed_in_source_order_and_keep_duplicates() -> None:
    settings = Settings(
        chunking_mode="recursive", chunk_max_chars=10, chunk_min_chars=1, chunk_overlap_chars=0
    )

    chunks = chunk_source("alpha beta alpha beta", ".txt", SOURCE, settings)

    assert chunks == [
        Chunk(
            text="alpha beta",
            source=SOURCE,
            chunk_index=index,
            chunking_strategy="recursive",
            chunk_id=chunks[index].chunk_id,
            chunk_type="recursive",
            oversized=False,
        )
        for index in range(2)
    ]
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_recursive_emits_an_oversized_atomic_token_intact_and_marks_it() -> None:
    settings = Settings(
        chunking_mode="recursive", chunk_max_chars=4, chunk_min_chars=1, chunk_overlap_chars=0
    )

    [chunk] = chunk_source("abcdefghij", ".txt", SOURCE, settings)

    assert (chunk.text, chunk.oversized) == ("abcdefghij", True)


@pytest.mark.parametrize(
    ("changed_source", "changed_mode"),
    [
        pytest.param(SOURCE, "fixed", id="same-input"),
        pytest.param("/docs/other.md", "fixed", id="other-source"),
        pytest.param(SOURCE, "recursive", id="other-strategy"),
    ],
)
def test_chunk_ids_are_deterministic_and_change_with_source_or_strategy(
    changed_source: str, changed_mode: str
) -> None:
    baseline = chunk_source("same words", ".txt", SOURCE, Settings(chunking_mode="fixed"))

    rerun = chunk_source(
        "same words",
        ".txt",
        changed_source,
        Settings(chunking_mode=changed_mode, chunk_min_chars=1),
    )

    same_input = (changed_source, changed_mode) == (SOURCE, "fixed")
    assert (baseline[0].chunk_id == rerun[0].chunk_id) is same_input
