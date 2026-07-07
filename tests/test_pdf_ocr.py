from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localrag.ingestion.parsers import pdf as pdf_module
from localrag.settings import Settings


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages


class _FakeBitmap:
    def __init__(self) -> None:
        self.closed = False

    def to_pil(self) -> object:
        return object()

    def close(self) -> None:
        self.closed = True


class _FakePdfPage:
    def __init__(self, size: tuple[float, float] = (595.0, 842.0)) -> None:
        self.closed = False
        self.last_bitmap: _FakeBitmap | None = None
        self.last_scale: float | None = None
        self._size = size

    def get_size(self) -> tuple[float, float]:
        return self._size

    def render(self, scale: float = 2.0) -> _FakeBitmap:
        self.last_scale = scale
        self.last_bitmap = _FakeBitmap()
        return self.last_bitmap

    def close(self) -> None:
        self.closed = True


class _FakePdfDocument:
    def __init__(self, page_count: int, page_size: tuple[float, float] = (595.0, 842.0)) -> None:
        self._pages = [_FakePdfPage(size=page_size) for _ in range(page_count)]
        self.closed = False

    def __getitem__(self, index: int) -> _FakePdfPage:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


def _settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


def test_parse_pdf_uses_text_layer_when_long_enough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("x" * 50)]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: _FakePdfDocument(1))

    def _fail_ocr(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("OCR should not run when the text layer is long enough")

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _fail_ocr)

    assert pdf_module.parse_pdf(path) == "x" * 50


def test_parse_pdf_falls_back_to_ocr_when_text_layer_too_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(1)
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("")]))
    monkeypatch.setattr(
        pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20, ocr_language="eng")
    )
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: fake_doc)
    monkeypatch.setattr(
        pdf_module.pytesseract, "image_to_string", lambda _image, lang: f"OCR:{lang}"
    )

    assert pdf_module.parse_pdf(path) == "OCR:eng"
    assert fake_doc.closed


def test_parse_pdf_skips_ocr_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("")]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_enabled=False))

    def _fail_open(*_args: Any, **_kwargs: Any) -> _FakePdfDocument:
        raise AssertionError("PdfDocument should not be opened when OCR is disabled")

    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", _fail_open)

    assert pdf_module.parse_pdf(path) == ""


def test_parse_pdf_ocr_uses_default_scale_for_normal_page_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(1, page_size=(595.0, 842.0))  # A4
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("")]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: fake_doc)
    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", lambda _image, **_kw: "text")

    pdf_module.parse_pdf(path)

    assert fake_doc[0].last_scale == 2.0


def test_parse_pdf_ocr_caps_scale_for_oversized_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    # ~33.1 x 23.4 inches (A0-scale architectural sheet), in PDF points.
    fake_doc = _FakePdfDocument(1, page_size=(2383.0, 1684.0))
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("")]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: fake_doc)
    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", lambda _image, **_kw: "text")

    pdf_module.parse_pdf(path)

    scale = fake_doc[0].last_scale
    assert scale is not None
    assert scale < 2.0
    # The capped render must not exceed the max dimension (2200px) on the longest side.
    assert 2383.0 * scale <= 2200.0 + 1e-6


def test_parse_pdf_ocr_closes_page_and_bitmap_after_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(1)
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("")]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: fake_doc)
    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", lambda _image, **_kw: "text")

    pdf_module.parse_pdf(path)

    page = fake_doc[0]
    assert page.closed
    assert page.last_bitmap is not None
    assert page.last_bitmap.closed


def test_parse_pdf_ocr_closes_page_even_when_ocr_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(1)
    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("")]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: fake_doc)

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _raise)

    pdf_module.parse_pdf(path)

    assert fake_doc[0].closed


def test_parse_pdf_ocr_failure_falls_back_to_text_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pdf_module, "PdfReader", lambda _: _FakeReader([_FakePage("short")]))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: _FakePdfDocument(1))

    def _raise(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _raise)

    assert pdf_module.parse_pdf(path) == "short"
