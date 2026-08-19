# -*- coding: utf-8 -*-
"""Tests for PDF page rendering and citation highlighting engine."""
from pathlib import Path
from ui.pdf_highlighter import get_pdf_path, render_highlighted_pdf_page, GUIDELINE_PDF_MAP

def test_get_pdf_path_resolves_all_guidelines():
    """Verify that get_pdf_path locates source PDFs for ESVS, NICE, SVS, and USPSTF."""
    guideline_keys = ["ESVS", "NICE", "SVS", "USPSTF"]
    for key in guideline_keys:
        path = get_pdf_path(key)
        assert path is not None, f"Failed to resolve path for {key}"
        assert path.exists(), f"Resolved path for {key} does not exist: {path}"

def test_render_highlighted_pdf_page_returns_png_bytes():
    """Verify that render_highlighted_pdf_page returns non-empty PNG bytes."""
    png_bytes = render_highlighted_pdf_page("ESVS", page_number=1, excerpt="Clinical Practice Guidelines")
    assert png_bytes is not None
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000
    # PNG signature check \x89PNG
    assert png_bytes.startswith(b"\x89PNG")

def test_render_highlighted_pdf_page_all_four_guidelines():
    """Verify page rendering works across all 4 guidelines."""
    for key in ["ESVS", "NICE", "SVS", "USPSTF"]:
        png = render_highlighted_pdf_page(key, page_number=1, excerpt="Abdominal Aortic Aneurysm")
        assert png is not None
        assert len(png) > 1000

def test_render_invalid_page_bounds_returns_valid_page():
    """Out of bound page number should fall back to valid page instead of raising error."""
    png = render_highlighted_pdf_page("USPSTF", page_number=999)
    assert png is not None
    assert png.startswith(b"\x89PNG")
