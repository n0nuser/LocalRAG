# Document formats

`localrag/ingestion/loader.py` selects a parser per file, and the parser
determines how the text is later chunked. Formats that produce Markdown are
chunked on heading structure; everything else uses the configured chunking mode.

## Supported formats

| Group | Extensions | Parser | Notes |
| --- | --- | --- | --- |
| Markdown | `.md`, `.markdown` | `parsers/markdown.py` | Chunked by heading structure |
| PDF | `.pdf` | `parsers/pdf.py` | `pdf-inspector` emits Markdown; scanned pages fall back to OCR |
| Plain text | `.txt`, `.rst` | `parsers/text.py` | |
| Code | `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp`, `.css`, `.html`, `.json`, `.yaml`, `.yml`, `.toml`, `.sh` | `parsers/code.py` | Chunked on code boundaries |
| Office and documents | `.doc`, `.docm`, `.docx`, `.odt`, `.rtf`, `.epub`, `.ppt`, `.pptm`, `.pptx`, `.pps`, `.ppsm`, `.ppsx`, `.pot`, `.odp`, `.xls`, `.xlsb`, `.xlsm`, `.xlsx`, `.ods`, `.csv` | `parsers/anydoc.py` | Converted to Markdown, then chunked structurally |

## anydoc conversion

`firecrawl-anydoc` is a **required** dependency, not an optional extra: office,
OpenDocument, EPUB, RTF, spreadsheet, and CSV files all go through
`anydoc.to_markdown()`. Because the output is Markdown, these documents get the
same heading-aware chunking as Markdown and PDF sources rather than being
flattened into a single text blob.

Format detection reads **file content**, not just the extension
(`anydoc.format_from_bytes`), so a mislabeled or extensionless file is still
routed correctly. CSV is the exception — it has no reliable content signature, so
the filename is used as a hint via `anydoc.format_from_path`.

A file is accepted when its suffix is in the supported set **or** content
detection recognizes it, which is why `is_supported()` can accept files with no
familiar extension.

## Unsupported files are rejected, not guessed at

`parse_file()` raises `UnsupportedFileTypeError` when no parser matches, instead
of falling back to the text parser. It also raises for binary content in a file
whose extension is textual (`.txt`, `.rst`), detected by sniffing the first 8 KB
for NUL bytes and undecodable UTF-8.

This matters because the previous catch-all fallback decoded arbitrary bytes into
chunks that were embedded and became retrievable — a corrupt corpus with no error
raised anywhere. Ingestion catches the error per file, so one unparseable input is
reported and skipped rather than aborting the batch.

## PDF and OCR

PDFs keep their own path: `pdf-inspector` extracts per-page Markdown and flags
pages whose text is unreliable, and only those pages are rasterized and run
through Tesseract. anydoc is not involved in PDF handling. See [ocr.md](ocr.md)
for the OCR settings, Tesseract install, and the heading-demotion behavior, and
[ADR 010](adr/010-pdf-inspector-extraction.md) for the extraction decision.

## Chunking

Which parser ran decides the chunking strategy: Markdown-producing formats are
split on headings, tables, and code blocks with `heading_path` metadata
preserved, while other formats use `CHUNKING_MODE`. See
[rag-retrieval.md](rag-retrieval.md) and
[ADR 021](adr/021-chunking-strategy-contract.md).
