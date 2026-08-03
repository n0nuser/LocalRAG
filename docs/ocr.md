# PDF OCR

LocalRAG falls back to **OCR** for scanned/image-only PDF pages during ingestion. **`pdf-inspector`** extracts each page's Markdown (headings, tables, lists) first and reports, per page, whether the text is unreliable (`needs_ocr`). A page is sent through OCR when `pdf-inspector` flags it as `needs_ocr`, **or** its extracted Markdown is shorter than `OCR_MIN_CHARS_PER_PAGE`; OCR renders the page with **pypdfium2** and reads it with **Tesseract** via `pytesseract`. This is implemented in `localrag/ingestion/parsers/pdf.py`.

`pdf-inspector` returns Markdown rather than plain text, so headings, tables, and multi-column reading order survive extraction instead of being flattened. Note that `chunk_document()` dispatches on file extension and `.pdf` is not in `MARKDOWN_EXTENSIONS`, so PDF chunks currently still take the non-Markdown path in `localrag/ingestion/structural_chunker.py`; routing them to the Markdown chunker is follow-up work (see [ADR 010](adr/010-pdf-inspector-extraction.md)).

Tesseract is a separate binary—not a Python package—so it must be installed on whatever host or container runs ingestion.

## Settings

| Env var | Default | Meaning |
| --- | --- | --- |
| `OCR_ENABLED` | `true` | Set to `false` to disable OCR entirely; scanned pages then yield whatever `pdf-inspector` extracted (empty string for pages it flags as `needs_ocr`), as before this feature existed. |
| `OCR_LANGUAGE` | `eng` | Tesseract language code (`ollama`-style tag list: `tesseract --list-langs`). Install the matching `tesseract-ocr-<lang>` package for non-English text. |
| `OCR_MIN_CHARS_PER_PAGE` | `20` | Pages whose `pdf-inspector` Markdown is shorter than this are treated as scanned and sent through OCR, even if `pdf-inspector` did not flag them as `needs_ocr`. |

## Installing Tesseract

- **Debian/Ubuntu (and this project's Docker image):** `apt-get install tesseract-ocr` (add `tesseract-ocr-<lang>` for extra languages, e.g. `tesseract-ocr-spa`).
- **macOS:** `brew install tesseract`.
- **Windows:** see the [Tesseract wiki install guide](https://github.com/tesseract-ocr/tesseract/blob/main/INSTALL.md).

If `tesseract` is missing from `PATH`, OCR fails silently per page (logged as a warning) and ingestion keeps whatever Markdown `pdf-inspector` extracted—ingestion never fails because of a missing OCR binary.

## Docker

The provided `Dockerfile` installs `tesseract-ocr` (English) and `tesseract-ocr-spa` (Spanish) in the base image. To OCR other languages inside Docker, add the relevant `tesseract-ocr-<lang>` package to the `apt-get install` line and set `OCR_LANGUAGE` accordingly.
