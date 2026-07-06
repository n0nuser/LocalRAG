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
    def to_pil(self) -> object:
        return object()


class _FakePdfPage:
    def render(self, scale: float = 2.0) -> _FakeBitmap:
        return _FakeBitmap()


class _FakePdfDocument:
    def __init__(self, page_count: int) -> None:
        self._pages = [_FakePdfPage() for _ in range(page_count)]
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
