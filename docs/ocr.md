# PDF OCR

LocalRAG falls back to **OCR** for scanned/image-only PDF pages during ingestion. `pypdf` extracts each page's text layer first; any page whose extracted text is shorter than `OCR_MIN_CHARS_PER_PAGE` is rasterized with **pypdfium2** and read with **Tesseract** via `pytesseract`. This is implemented in `localrag/ingestion/parsers/pdf.py`.

Tesseract is a separate binary—not a Python package—so it must be installed on whatever host or container runs ingestion.

## Settings

| Env var | Default | Meaning |
| --- | --- | --- |
| `OCR_ENABLED` | `true` | Set to `false` to disable OCR entirely; scanned pages then yield empty text, as before this feature existed. |
| `OCR_LANGUAGE` | `eng` | Tesseract language code (`ollama`-style tag list: `tesseract --list-langs`). Install the matching `tesseract-ocr-<lang>` package for non-English text. |
| `OCR_MIN_CHARS_PER_PAGE` | `20` | Pages with a `pypdf` text layer shorter than this are treated as scanned and sent through OCR. |

## Installing Tesseract

- **Debian/Ubuntu (and this project's Docker image):** `apt-get install tesseract-ocr` (add `tesseract-ocr-<lang>` for extra languages, e.g. `tesseract-ocr-spa`).
- **macOS:** `brew install tesseract`.
- **Windows:** see the [Tesseract wiki install guide](https://github.com/tesseract-ocr/tesseract/blob/main/INSTALL.md).

If `tesseract` is missing from `PATH`, OCR fails silently per page (logged as a warning) and ingestion keeps whatever text `pypdf` extracted—ingestion never fails because of a missing OCR binary.

## Docker

The provided `Dockerfile` installs `tesseract-ocr` (English) and `tesseract-ocr-spa` (Spanish) in the base image. To OCR other languages inside Docker, add the relevant `tesseract-ocr-<lang>` package to the `apt-get install` line and set `OCR_LANGUAGE` accordingly.
