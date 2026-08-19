# -*- coding: utf-8 -*-
"""Forwarding module for PDF highlighter to maintain compatibility across layers."""
from ui.pdf_highlighter import (
    GUIDELINE_PDF_MAP,
    get_pdf_path,
    render_highlighted_pdf_page,
)

__all__ = [
    "GUIDELINE_PDF_MAP",
    "get_pdf_path",
    "render_highlighted_pdf_page",
]
