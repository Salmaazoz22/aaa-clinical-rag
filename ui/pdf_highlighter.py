# -*- coding: utf-8 -*-
"""PDF Page Screenshot & Citation Highlight Engine.

Renders PDF pages from any of the 4 clinical guidelines (ESVS 2024, NICE NG156,
SVS 2018, USPSTF 2019) to PNG images with yellow/amber bounding box highlights
placed over cited text/paragraphs.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
import pymupdf

# Canonical mapping of document names, IDs, and source filenames to PDF files in data/pdfs/
GUIDELINE_PDF_MAP: dict[str, str] = {
    "ESVS": "ESVS_2024_AAA_Guidelines.pdf",
    "ESVS_2024": "ESVS_2024_AAA_Guidelines.pdf",
    "ESVS_2024_AAA_Guidelines.pdf": "ESVS_2024_AAA_Guidelines.pdf",
    "SVS": "SVS_Guideline_AAA_Slides_0.pdf",
    "SVS_2018": "SVS_Guideline_AAA_Slides_0.pdf",
    "SVS_Guideline_AAA_Slides_0.pdf": "SVS_Guideline_AAA_Slides_0.pdf",
    "USPSTF": "abdom-aortic-aneurysm-screening-final-rs.pdf",
    "USPSTF_2019": "abdom-aortic-aneurysm-screening-final-rs.pdf",
    "abdom-aortic-aneurysm-screening-final-rs.pdf": "abdom-aortic-aneurysm-screening-final-rs.pdf",
    "NICE": "abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf",
    "NICE_2020": "abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf",
    "NG156": "abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf",
    "abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf": (
        "abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf"
    ),
}

DEFAULT_PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def get_pdf_path(source_doc: str, pdf_dir: Path | str | None = None) -> Path | None:
    """Resolve the absolute path to a guideline PDF file."""
    base_dir = Path(pdf_dir) if pdf_dir else DEFAULT_PDF_DIR
    clean_name = str(source_doc).strip()
    filename = GUIDELINE_PDF_MAP.get(clean_name) or clean_name
    
    if not filename.endswith(".pdf"):
        filename += ".pdf"
        
    path = base_dir / filename
    if path.exists():
        return path
        
    # Fallback search by key substring
    for key, mapped in GUIDELINE_PDF_MAP.items():
        if key.lower() in clean_name.lower():
            candidate = base_dir / mapped
            if candidate.exists():
                return candidate
                
    return None


@functools.lru_cache(maxsize=128)
def render_highlighted_pdf_page(
    source_doc: str,
    page_number: int,
    excerpt: str | None = None,
    dpi: int = 130,
    pdf_dir: str | None = None,
) -> bytes | None:
    """Render a PDF page to PNG bytes with highlight bounding boxes on matching excerpt text.

    Args:
        source_doc: Document ID, name, or filename (e.g. 'ESVS_2024', 'NICE', 'SVS_Guideline_AAA_Slides_0.pdf').
        page_number: 1-indexed page number in the PDF.
        excerpt: Optional chunk text or excerpt to search and highlight on the page.
        dpi: Resolution DPI for rendered PNG (default 130 DPI for crisp readability).
        pdf_dir: Optional custom PDF directory path.

    Returns:
        PNG image bytes, or None if PDF/page cannot be loaded.
    """
    pdf_path = get_pdf_path(source_doc, pdf_dir=pdf_dir)
    if not pdf_path or not pdf_path.exists():
        return None

    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return None

    if len(doc) == 0:
        return None

    # Convert 1-indexed page number to 0-indexed page index safely
    page_idx = max(0, min(page_number - 1, len(doc) - 1))
    page = doc[page_idx]

    # Perform text search and highlighting if excerpt is provided
    if excerpt and excerpt.strip():
        rects = _find_excerpt_rects(page, excerpt)
        for r in rects:
            try:
                annot = page.add_highlight_annot(r)
                # Golden yellow highlight: RGB (1.0, 0.82, 0.1)
                annot.set_colors(stroke=(1.0, 0.82, 0.1))
                annot.update()
            except Exception:
                pass

    # Render page to PNG pixmap
    try:
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        return png_bytes
    except Exception:
        return None


def _find_excerpt_rects(page: pymupdf.Page, excerpt: str) -> list[pymupdf.Rect]:
    """Locate text bounding boxes for excerpt on the given page."""
    rects: list[pymupdf.Rect] = []
    clean_excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if not clean_excerpt:
        return rects

    # Strategy 1: Try exact search of the first 60 chars of excerpt
    first_phrase = clean_excerpt[:60].strip()
    if len(first_phrase) > 10:
        found = page.search_for(first_phrase)
        if found:
            rects.extend(found)

    # Strategy 2: Split into sentences and search each sentence
    if not rects:
        sentences = [s.strip() for s in re.split(r"[.\n;]", clean_excerpt) if len(s.strip()) > 15]
        for sentence in sentences[:4]:
            found = page.search_for(sentence)
            if found:
                rects.extend(found)

    # Strategy 3: Significant word n-grams (windows of 3-5 distinct words)
    if not rects:
        words = [w for w in clean_excerpt.split() if len(w) >= 4 and not w.isdigit()]
        for i in range(0, max(1, len(words) - 3), 3):
            phrase = " ".join(words[i : i + 3])
            if len(phrase) >= 10:
                found = page.search_for(phrase)
                if found:
                    rects.extend(found)

    return rects
