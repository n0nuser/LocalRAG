from __future__ import annotations

import logging
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract
from pypdf import PageObject, PdfReader

from localrag.settings import Settings, get_settings

logger = logging.getLogger(__name__)


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
    try:
        bitmap = ocr_doc[index].render(scale=2.0)
        image = bitmap.to_pil()
        return pytesseract.image_to_string(image, lang=language).strip()
    except Exception:
        logger.warning("ocr_page_failed page=%d", index, exc_info=True)
        return ""
