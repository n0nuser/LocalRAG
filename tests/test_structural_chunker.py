from __future__ import annotations

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


def test_chunk_document_non_markdown_packs_paragraphs() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    settings = Settings(chunk_max_chars=40, chunk_min_chars=20)

    chunks = chunk_document(text, ".txt", settings)

    assert len(chunks) == 2
    assert chunks[0].chunk_type == "text_block"
    assert chunks[0].heading_path == ""
