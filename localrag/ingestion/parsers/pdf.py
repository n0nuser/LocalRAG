from __future__ import annotations

import gc
import logging
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract
from pypdf import PageObject, PdfReader

from localrag.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_OCR_DEFAULT_SCALE = 2.0
# Absolute cap on the rendered bitmap's longest side, in pixels. Urban-planning /
# architectural PDFs routinely embed oversized sheets (A0/A1, or larger) as a
# single page; at the default scale those pages can be tens of megapixels each,
# and across a multi-hundred-page scanned document that's enough to exhaust
# container memory. This bounds per-page memory regardless of the page's own
# physical size, at the cost of lower OCR resolution on oversized pages only.
_OCR_MAX_DIMENSION_PX = 2200.0


def parse_pdf(path: Path) -> str:
    settings = get_settings()
    reader = PdfReader(str(path))
    ocr_doc = pdfium.PdfDocument(str(path)) if settings.ocr_enabled else None
    try:
        parts = [
            _extract_page_text(page, index, ocr_doc, settings)
            for index, page in enumerate(reader.pages)
        ]
    finally:
        if ocr_doc is not None:
            ocr_doc.close()
    return "\n".join(parts).strip()


def _extract_page_text(
    page: PageObject,
    index: int,
    ocr_doc: pdfium.PdfDocument | None,
    settings: Settings,
) -> str:
    text = (page.extract_text() or "").strip()
    if ocr_doc is None or len(text) >= settings.ocr_min_chars_per_page:
        return text
    ocr_text = _ocr_page(ocr_doc, index, settings.ocr_language)
    return ocr_text or text


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
