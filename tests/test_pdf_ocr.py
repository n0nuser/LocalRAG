from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localrag.ingestion.parsers import pdf as pdf_module
from localrag.settings import Settings


class _FakePageMarkdown:
    def __init__(self, page: int, markdown: str, *, needs_ocr: bool = False) -> None:
        self.page = page
        self.markdown = markdown
        self.needs_ocr = needs_ocr


class _FakePagesExtractionResult:
    def __init__(self, pages: list[_FakePageMarkdown]) -> None:
        self.pages = pages
        self.pages_with_tables: list[int] = []
        self.pages_with_columns: list[int] = []
        self.pages_needing_ocr: list[int] = []
        self.is_complex = False


class _FakePdfInspector:
    def __init__(self, result: _FakePagesExtractionResult | Exception) -> None:
        self._result = result

    def extract_pages_markdown(self, _path: str, pages: list[int] | None = None) -> Any:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


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


def _patch_inspector(monkeypatch: pytest.MonkeyPatch, pages: list[_FakePageMarkdown]) -> None:
    monkeypatch.setattr(
        pdf_module,
        "pdf_inspector",
        _FakePdfInspector(_FakePagesExtractionResult(pages)),
    )


def test_parse_pdf_uses_text_layer_when_long_enough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "x" * 50)])
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: _FakePdfDocument(1))

    def _fail_ocr(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("OCR should not run when the text layer is long enough")

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _fail_ocr)

    assert pdf_module.parse_pdf(path) == "x" * 50


def test_parse_pdf_does_not_open_pdfium_document_when_no_page_needs_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Opening the pypdfium2 document costs time and native memory.

    Text-layer PDFs are the common case and must not pay for it.
    """
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "x" * 50), _FakePageMarkdown(1, "y" * 50)])
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))

    def _fail_open(*_args: Any, **_kwargs: Any) -> object:
        raise AssertionError("pdfium document must not be opened when no page needs OCR")

    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", _fail_open)

    assert pdf_module.parse_pdf(path) == "x" * 50 + "\n" + "y" * 50


def test_parse_pdf_opens_pdfium_document_once_across_multiple_ocr_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lazily-opened document is reused, not reopened per OCR page."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(3)
    opens: list[str] = []

    def _open(arg: str) -> _FakePdfDocument:
        opens.append(arg)
        return fake_doc

    _patch_inspector(
        monkeypatch,
        [
            _FakePageMarkdown(0, "", needs_ocr=True),
            _FakePageMarkdown(1, "x" * 50),
            _FakePageMarkdown(2, "", needs_ocr=True),
        ],
    )
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", _open)
    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", lambda _image, **_kw: "OCR")

    pdf_module.parse_pdf(path)

    assert len(opens) == 1
    assert fake_doc.closed is True


def test_parse_pdf_falls_back_to_ocr_when_text_layer_too_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(1)
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "")])
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

    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "")])
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
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "")])
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
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "")])
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
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "")])
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
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "")])
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

    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, "short")])
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: _FakePdfDocument(1))

    def _raise(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _raise)

    assert pdf_module.parse_pdf(path) == "short"


def test_parse_pdf_returns_inspector_markdown_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    markdown = "# Heading\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n- item one\n- item two"
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, markdown)])
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=5))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: _FakePdfDocument(1))

    def _fail_ocr(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("OCR should not run for long, non-flagged markdown")

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _fail_ocr)

    assert pdf_module.parse_pdf(path) == markdown


def test_parse_pdf_needs_ocr_flag_triggers_ocr_even_when_markdown_is_long(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    long_markdown = "x" * 500
    _patch_inspector(monkeypatch, [_FakePageMarkdown(0, long_markdown, needs_ocr=True)])
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: _FakePdfDocument(1))
    monkeypatch.setattr(
        pdf_module.pytesseract, "image_to_string", lambda _image, **_kw: "OCR result"
    )

    assert pdf_module.parse_pdf(path) == "OCR result"


def test_parse_pdf_routes_ocr_using_zero_indexed_page_not_one_indexed_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PageMarkdown.page is 0-indexed; pages_needing_ocr is 1-indexed.

    Only page index 2 (0-indexed) needs OCR. Assert tesseract receives that
    exact pdfium page object, catching an off-by-one between the two
    conventions the library mixes.
    """
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    fake_doc = _FakePdfDocument(4)
    pages = [
        _FakePageMarkdown(0, "x" * 50),
        _FakePageMarkdown(1, "x" * 50),
        _FakePageMarkdown(2, "", needs_ocr=True),
        _FakePageMarkdown(3, "x" * 50),
    ]
    _patch_inspector(monkeypatch, pages)
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_min_chars_per_page=20))
    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", lambda _: fake_doc)

    ocr_calls: list[object] = []

    def _record(image: object, **_kw: object) -> str:
        ocr_calls.append(image)
        return "OCR"

    monkeypatch.setattr(pdf_module.pytesseract, "image_to_string", _record)

    pdf_module.parse_pdf(path)

    # Only page index 2 should have been rendered/OCR'd.
    assert fake_doc[0].last_bitmap is None
    assert fake_doc[1].last_bitmap is None
    assert fake_doc[2].last_bitmap is not None
    assert fake_doc[3].last_bitmap is None
    assert len(ocr_calls) == 1


def test_parse_pdf_demotes_running_headers_repeated_across_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Font-size heading inference promotes mastheads and footers to headings.

    They repeat on every page; real section titles do not. Demoting them keeps
    them out of downstream chunk heading paths.
    """
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    pages = [
        _FakePageMarkdown(index, f"# Boletin Oficial\n\n## Seccion {index}\n\nCuerpo {index}.")
        for index in range(4)
    ]
    _patch_inspector(monkeypatch, pages)
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_enabled=False))

    out = pdf_module.parse_pdf(path)

    # The masthead survives as text but no longer as a heading.
    assert "# Boletin Oficial" not in out
    assert "Boletin Oficial" in out
    # Genuine per-page section headings are untouched.
    assert "## Seccion 0" in out
    assert "## Seccion 3" in out


def test_parse_pdf_keeps_repeated_headings_in_short_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In a 2-page document a twice-seen heading is not evidence of furniture."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    pages = [_FakePageMarkdown(index, "# Repeated Title\n\nBody text here.") for index in range(2)]
    _patch_inspector(monkeypatch, pages)
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings(ocr_enabled=False))

    assert pdf_module.parse_pdf(path).count("# Repeated Title") == 2


def test_parse_pdf_whole_document_extraction_failure_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pdf_module, "pdf_inspector", _FakePdfInspector(ValueError("corrupt PDF")))
    monkeypatch.setattr(pdf_module, "get_settings", lambda: _settings())

    def _fail_open(*_args: Any, **_kwargs: Any) -> _FakePdfDocument:
        raise AssertionError("PdfDocument should not be opened when extraction fails")

    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", _fail_open)

    assert pdf_module.parse_pdf(path) == ""
