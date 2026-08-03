from __future__ import annotations

import gc
import logging
import re
from collections import Counter
from pathlib import Path

import pdf_inspector
import pypdfium2 as pdfium
import pytesseract

from localrag.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
# pdf_inspector infers heading levels from font-size ratios, so a page's running
# header or footer (masthead, "Núm. 31 ... Pág. 6539", an ISSN line) is often set
# in a larger face than body text and comes back as a heading. Those repeat on
# every page; real section titles do not. A line that appears as a heading on
# more than this fraction of a multi-page document's pages is treated as furniture
# and demoted to plain text, so it stops polluting chunk heading paths.
_REPEATED_HEADING_PAGE_RATIO = 0.5
# Below this page count "repeats on most pages" is not evidence of anything —
# a 2-page document would demote a heading that legitimately appears twice.
_REPEATED_HEADING_MIN_PAGES = 3

_OCR_DEFAULT_SCALE = 2.0
# Absolute cap on the rendered bitmap's longest side, in pixels. Urban-planning /
# architectural PDFs routinely embed oversized sheets (A0/A1, or larger) as a
# single page; at the default scale those pages can be tens of megapixels each,
# and across a multi-hundred-page scanned document that's enough to exhaust
# container memory. This bounds per-page memory regardless of the page's own
# physical size, at the cost of lower OCR resolution on oversized pages only.
_OCR_MAX_DIMENSION_PX = 2200.0


def parse_pdf(path: Path) -> str:
    """Extract Markdown text from a PDF, OCR-ing pages whose text layer is unreliable.

    Uses ``pdf_inspector`` for per-page Markdown extraction (preserving headings,
    tables, and multi-column reading order), and falls back
    to rasterizing + Tesseract OCR for pages ``pdf_inspector`` flags as needing it,
    or whose extracted Markdown is shorter than ``settings.ocr_min_chars_per_page``.

    Args:
        path: Path to the PDF file.

    Returns:
        The document's Markdown text, pages joined by newlines and stripped.
        Empty string if the whole document fails to parse (logged as a warning).
    """
    settings = get_settings()
    try:
        result = pdf_inspector.extract_pages_markdown(str(path))
    except Exception:
        logger.warning("pdf_extraction_failed path=%s", path, exc_info=True)
        return ""

    # Opening the pypdfium2 document costs real time and native memory, so it is
    # deferred until a page actually needs OCR. Text-layer PDFs — the common case —
    # never pay for it.
    ocr_doc: pdfium.PdfDocument | None = None
    try:
        parts = []
        for page in result.pages:
            if not _page_needs_ocr(page, settings):
                parts.append(page.markdown)
                continue
            if ocr_doc is None:
                ocr_doc = pdfium.PdfDocument(str(path))
            ocr_text = _ocr_page(ocr_doc, page.page, settings.ocr_language)
            parts.append(ocr_text or page.markdown)
    finally:
        if ocr_doc is not None:
            ocr_doc.close()
    return "\n".join(_demote_repeated_headings(parts)).strip()


def _demote_repeated_headings(pages: list[str]) -> list[str]:
    """Turn headings that repeat across most pages back into plain text.

    Args:
        pages: Per-page Markdown, in document order.

    Returns:
        The pages with running headers/footers demoted, leaving real section
        headings untouched. Returned unchanged for short documents, where
        repetition carries no signal.
    """
    if len(pages) < _REPEATED_HEADING_MIN_PAGES:
        return pages

    pages_per_heading: Counter[str] = Counter()
    for page in pages:
        headings = {
            match.group(2).strip()
            for line in page.splitlines()
            if (match := _HEADING_PATTERN.match(line.strip()))
        }
        pages_per_heading.update(headings)

    threshold = len(pages) * _REPEATED_HEADING_PAGE_RATIO
    repeated = {text for text, count in pages_per_heading.items() if count > threshold}
    if not repeated:
        return pages

    logger.debug("demoted_repeated_pdf_headings count=%d", len(repeated))
    return [_demote_headings_in_page(page, repeated) for page in pages]


def _demote_headings_in_page(page: str, repeated: set[str]) -> str:
    lines = []
    for line in page.splitlines():
        match = _HEADING_PATTERN.match(line.strip())
        if match and match.group(2).strip() in repeated:
            lines.append(match.group(2).strip())
            continue
        lines.append(line)
    return "\n".join(lines)


def _page_needs_ocr(page: pdf_inspector.PageMarkdown, settings: Settings) -> bool:
    """Report whether a page's extracted Markdown is too weak to trust as-is."""
    if not settings.ocr_enabled:
        return False
    return page.needs_ocr or len(page.markdown) < settings.ocr_min_chars_per_page


def _ocr_page(ocr_doc: pdfium.PdfDocument, index: int, language: str) -> str:
    # pypdfium2 pages/bitmaps wrap native memory that isn't reclaimed by Python's
    # GC — without explicit close(), a large scanned PDF accumulates one rendered
    # page's worth of native memory per page for the life of the document.
    page = ocr_doc[index]
    try:
        scale = _render_scale_for(page)
        bitmap = page.render(scale=scale)
        try:
            image = bitmap.to_pil()
            return pytesseract.image_to_string(image, lang=language).strip()
        finally:
            bitmap.close()
    except Exception:
        logger.warning("ocr_page_failed page=%d", index, exc_info=True)
        return ""
    finally:
        page.close()
        # pypdfium2's ctypes-backed objects can hold their native buffer alive
        # until Python's cyclic GC actually runs, even after close() — without
        # forcing a collection here, a long scanned document accumulates native
        # memory across pages regardless of per-object cleanup.
        gc.collect()


def _render_scale_for(page: pdfium.PdfPage) -> float:
    width_pt, height_pt = page.get_size()
    longest_pt = max(width_pt, height_pt)
    if longest_pt <= 0:
        return _OCR_DEFAULT_SCALE
    longest_px_at_default = longest_pt * _OCR_DEFAULT_SCALE
    if longest_px_at_default <= _OCR_MAX_DIMENSION_PX:
        return _OCR_DEFAULT_SCALE
    return _OCR_MAX_DIMENSION_PX / longest_pt
