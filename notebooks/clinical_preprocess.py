# -*- coding: utf-8 -*-
"""AAA clinical PDF preprocessing.

Rebuilds extraction, cleaning, recommendation capture, and document metadata
without inventing clinical content. Original PDFs are never modified.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import fitz
except ImportError as exc:
    raise ImportError("PyMuPDF is required (import fitz / pymupdf).") from exc

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    pdfplumber = None
    HAS_PDFPLUMBER = False

try:
    import pytesseract
    from PIL import Image

    HAS_OCR = True
except ImportError:
    pytesseract = None
    Image = None
    HAS_OCR = False

logger = logging.getLogger("aaa_clinical_preprocess")

PREFERRED_PDF_DIRS = ("data/pdfs", "data/raw", "data", "documents", "docs", "pdfs")
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".ipynb_checkpoints",
    "processed",
    "chunks",
    "embeddings",
}

DOC_ID_NICE = "NICE_NG156"
DOC_ID_USPSTF = "USPSTF_2019"
DOC_ID_SVS = "SVS_2018"
DOC_ID_ESVS = "ESVS_2024"
DOC_ID_OTHER_PREFIX = "OTHER"

LOW_TEXT_CHAR_THRESHOLD = 80
GE = "\u2265"
LE = "\u2264"
EN_DASH = "\u2013"
MINUS_SIGN = "\u2212"
BULLET = "\u2022"
REPLACEMENT = "\ufffd"

PROCESSED_OUTPUT_NAMES = {
    "pages_df.parquet",
    "pages.json",
    "document_metadata.json",
    "recommendations.json",
    "extraction_report.json",
}

PAGE_COLUMNS = [
    "document_id",
    "document_name",
    "document_type",
    "source_file",
    "page_number",
    "section_title",
    "section_source",
    "raw_text",
    "clean_text",
    "table_text",
    "character_count",
    "word_count",
    "line_count",
    "extraction_status",
    "extraction_library",
]


# ---------------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    candidates = [cwd, *cwd.parents]
    if cwd.name == "notebooks":
        candidates.insert(0, cwd.parent)

    def looks_like_root(path: Path) -> bool:
        if (path / "notebooks").is_dir() and (path / "data").is_dir():
            return True
        return any((path / rel).exists() for rel in PREFERRED_PDF_DIRS)

    for cand in candidates:
        try:
            if looks_like_root(cand):
                return cand
        except OSError:
            continue
    return cwd


def _is_excluded(path: Path, project_root: Path) -> bool:
    try:
        parts = path.relative_to(project_root).parts
    except ValueError:
        parts = path.parts
    return any(part in EXCLUDED_DIR_NAMES for part in parts)


def discover_pdfs(project_root: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def add(pdf: Path) -> None:
        try:
            resolved = pdf.resolve()
        except OSError as exc:
            logger.warning("Skipping unreadable path %s (%s)", pdf, exc)
            return
        if resolved in seen or not resolved.is_file() or _is_excluded(resolved, project_root):
            return
        seen.add(resolved)
        found.append(resolved)

    for rel in PREFERRED_PDF_DIRS:
        folder = project_root / rel
        if folder.is_dir():
            for pdf in sorted(folder.rglob("*.pdf")):
                add(pdf)
    for pdf in sorted(project_root.rglob("*.pdf")):
        add(pdf)
    return found


def relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Document identification (filename first, then page-1 publisher strings)
# ---------------------------------------------------------------------------

_NICE_FILENAME_RE = re.compile(r"ng156|\bnice\b", re.I)
_USPSTF_FILENAME_RE = re.compile(r"uspstf", re.I)
_SVS_FILENAME_RE = re.compile(r"(^|[^a-z])svs([^a-z]|$)", re.I)
_ESVS_FILENAME_RE = re.compile(r"(^|[^a-z])esvs([^a-z]|$)", re.I)
_USPSTF_PAGE1_RE = re.compile(r"U\.?\s*S\.?\s+Preventive Services Task Force|\bUSPSTF\b", re.I)
_NICE_PAGE1_RE = re.compile(r"National Institute for Health and Care Excellence|\bNG156\b", re.I)
_SVS_PAGE1_RE = re.compile(r"Society for Vascular Surgery|\bSVS\b", re.I)
# Must be tested BEFORE _SVS_PAGE1_RE: "European Society for Vascular Surgery"
# contains "Society for Vascular Surgery" and would otherwise match SVS.
_ESVS_PAGE1_RE = re.compile(r"European Society for Vascular Surgery|\bESVS\b", re.I)

_PAGE1_CACHE: dict[str, str] = {}
_META_CACHE: dict[str, dict[str, Any]] = {}


def read_first_page_text(pdf_path: Path) -> str:
    key = str(pdf_path.resolve())
    if key in _PAGE1_CACHE:
        return _PAGE1_CACHE[key]
    text = ""
    try:
        doc = fitz.open(pdf_path)
        try:
            if doc.page_count:
                text = doc[0].get_text("text") or ""
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("Could not read page 1 of %s: %s", pdf_path.name, exc)
    _PAGE1_CACHE[key] = text
    return text


def pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    key = str(pdf_path.resolve())
    if key in _META_CACHE:
        return dict(_META_CACHE[key])
    try:
        doc = fitz.open(pdf_path)
        try:
            meta = dict(doc.metadata or {})
            meta["page_count"] = int(doc.page_count)
            _META_CACHE[key] = meta
            return dict(meta)
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("Could not read metadata for %s: %s", pdf_path.name, exc)
        return {}


def _slug_stem(pdf_path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_")
    return slug[:80] or "unnamed"


def _clean_pdf_title(title: str | None) -> str | None:
    if not title:
        return None
    title = title.strip()
    if not title:
        return None
    if title.lower() in {"(unspecified)", "unknown", "powerpoint presentation"}:
        return None
    return title


def identify_document(pdf_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    blob = pdf_path.name
    if project_root is not None:
        blob = f"{pdf_path.name} {relative_posix(pdf_path, project_root)}"
    meta = pdf_metadata(pdf_path)
    pdf_title = _clean_pdf_title(meta.get("title"))

    def rec(doc_id: str, name: str, dtype: str, method: str) -> dict[str, Any]:
        return {
            "document_id": doc_id,
            "document_name": name,
            "document_type": dtype,
            "source_file": pdf_path.name,
            "identification_method": method,
        }

    if _NICE_FILENAME_RE.search(blob):
        return rec(
            DOC_ID_NICE,
            pdf_title or "Abdominal aortic aneurysm: diagnosis and management",
            "official guideline",
            "filename",
        )
    if _USPSTF_FILENAME_RE.search(blob):
        return rec(
            DOC_ID_USPSTF,
            pdf_title or "Screening for Abdominal Aortic Aneurysm: US Preventive Services Task Force Recommendation Statement",
            "government/public health recommendation",
            "filename",
        )
    if _ESVS_FILENAME_RE.search(blob.lower()):
        return rec(
            DOC_ID_ESVS,
            pdf_title
            or "European Society for Vascular Surgery (ESVS) 2024 Clinical Practice Guidelines on the Management of Abdominal Aorto-Iliac Artery Aneurysms",
            "official guideline",
            "filename",
        )
    if _SVS_FILENAME_RE.search(blob.lower()):
        return rec(
            DOC_ID_SVS,
            "Care of Patients with an Abdominal Aortic Aneurysm",
            "official guideline",
            "filename",
        )

    page1 = read_first_page_text(pdf_path)
    if _USPSTF_PAGE1_RE.search(page1):
        return rec(
            DOC_ID_USPSTF,
            pdf_title or "Screening for Abdominal Aortic Aneurysm: US Preventive Services Task Force Recommendation Statement",
            "government/public health recommendation",
            "page1_content",
        )
    if _NICE_PAGE1_RE.search(page1):
        return rec(
            DOC_ID_NICE,
            pdf_title or "Abdominal aortic aneurysm: diagnosis and management",
            "official guideline",
            "page1_content",
        )
    if _ESVS_PAGE1_RE.search(page1):
        return rec(
            DOC_ID_ESVS,
            pdf_title
            or "European Society for Vascular Surgery (ESVS) 2024 Clinical Practice Guidelines on the Management of Abdominal Aorto-Iliac Artery Aneurysms",
            "official guideline",
            "page1_content",
        )
    if _SVS_PAGE1_RE.search(page1):
        heading = _first_nonempty_lines(page1, 2)
        return rec(
            DOC_ID_SVS,
            heading or "Care of Patients with an Abdominal Aortic Aneurysm",
            "official guideline",
            "page1_content",
        )

    name = pdf_title or pdf_path.stem
    dtype = "review article" if re.search(r"literature review", page1 + " " + (pdf_title or ""), re.I) else "other"
    author = (meta.get("author") or "").strip()
    if dtype == "other" and re.search(r"literature review", author, re.I):
        dtype = "review article"
    return rec(f"{DOC_ID_OTHER_PREFIX}_{_slug_stem(pdf_path)}", name, dtype, "other")


def _first_nonempty_lines(text: str, n: int = 2) -> str | None:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    return " ".join(lines[:n]).strip() or None


def assess_pdf_validity(pdf_path: Path) -> dict[str, Any]:
    """Return exclusion reason or None. Never deletes the file."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return {"valid": False, "reason": f"PDF could not be opened: {exc}"}
    try:
        n_pages = int(doc.page_count)
        if n_pages < 1:
            return {"valid": False, "reason": "PDF has zero pages"}
        sample = []
        for i in range(min(n_pages, 3)):
            sample.append(doc[i].get_text("text") or "")
        blob = "\n".join(sample)
        if not blob.strip():
            # Might still be a scanned AAA guideline; keep unless clearly unrelated later.
            return {"valid": True, "reason": None, "note": "no extractable text on first pages"}
        if not re.search(r"abdominal aortic aneurysm|\bAAA\b|aortic aneurysm", blob, re.I):
            # Check later pages before excluding.
            extra = []
            for i in range(n_pages):
                extra.append(doc[i].get_text("text") or "")
                if len("\n".join(extra)) > 8000:
                    break
            if not re.search(r"abdominal aortic aneurysm|\bAAA\b|aortic aneurysm", "\n".join(extra), re.I):
                return {"valid": False, "reason": "No AAA-related content detected in extractable text"}
        return {"valid": True, "reason": None}
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _blocks_to_text(page: "fitz.Page") -> str:
    try:
        blocks = page.get_text("blocks") or []
    except Exception as exc:
        logger.warning("blocks extraction failed: %s", exc)
        return ""
    lines: list[str] = []
    for block in sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1))):
        if len(block) < 5:
            continue
        txt = (block[4] or "").strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines)


def _dict_to_text(page: "fitz.Page") -> str:
    try:
        data = page.get_text("dict") or {}
    except Exception as exc:
        logger.warning("dict extraction failed: %s", exc)
        return ""
    out: list[str] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            parts = [span.get("text", "") for span in line.get("spans", [])]
            joined = "".join(parts).strip()
            if joined:
                out.append(joined)
    return "\n".join(out)


def _page_has_images(page: "fitz.Page") -> bool:
    try:
        return bool(page.get_images())
    except Exception:
        return False


def _looks_like_toc(text: str) -> bool:
    return len(re.findall(r"\.{5,}", text or "")) >= 4


def _looks_like_table(text: str) -> bool:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    pipe_rows = sum(1 for ln in lines if ln.count("|") >= 2)
    return pipe_rows >= 3 or (pipe_rows / max(len(lines), 1) >= 0.4)


def _non_alnum_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    alnum = sum(ch.isalnum() for ch in compact)
    return 1.0 - (alnum / len(compact))


def classify_extraction_status(text: str, has_images: bool = False) -> str:
    raw = text or ""
    stripped = raw.strip()
    n = len(stripped)
    if n == 0:
        return "ocr_required" if has_images else "image_only"
    if raw.count(REPLACEMENT) >= 12 and (raw.count(REPLACEMENT) / max(n, 1)) > 0.02:
        return "corrupted"
    if _looks_like_toc(raw) or _looks_like_table(raw):
        return "ok" if n >= LOW_TEXT_CHAR_THRESHOLD else "low_text"
    ratio = _non_alnum_ratio(raw)
    if ratio >= 0.65 and n >= 80 and not _looks_like_toc(raw):
        return "corrupted"
    if n < LOW_TEXT_CHAR_THRESHOLD:
        return "low_text"
    return "ok"


def maybe_ocr_page(page: "fitz.Page") -> str:
    if not HAS_OCR:
        return ""
    try:
        pix = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(image) or ""
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return ""


def extract_page_with_fallback(page: "fitz.Page") -> dict[str, Any]:
    warnings: list[str] = []
    has_images = _page_has_images(page)
    primary = page.get_text("text") or ""
    library = "pymupdf"
    chosen = primary
    status = classify_extraction_status(primary, has_images)

    if status in {"image_only", "ocr_required", "low_text", "corrupted"}:
        alt_blocks = _blocks_to_text(page)
        if len(alt_blocks.strip()) > len(chosen.strip()):
            chosen = alt_blocks
            library = "pymupdf"
            status = classify_extraction_status(chosen, has_images)
            warnings.append("used pymupdf blocks fallback")

    if status in {"image_only", "ocr_required", "low_text", "corrupted"}:
        alt_dict = _dict_to_text(page)
        if len(alt_dict.strip()) > len(chosen.strip()):
            chosen = alt_dict
            library = "pymupdf"
            status = classify_extraction_status(chosen, has_images)
            warnings.append("used pymupdf dict fallback")

    if status in {"image_only", "ocr_required"} and has_images:
        ocr_text = maybe_ocr_page(page)
        if ocr_text.strip():
            chosen = ocr_text
            library = "pymupdf+ocr"
            status = classify_extraction_status(chosen, has_images)
            warnings.append("OCR used because native text layer was empty")
        else:
            warnings.append("image-only/sparse page; OCR was not available or produced no text")

    return {
        "text": chosen,
        "extraction_library": library,
        "extraction_status": status,
        "extraction_warning": "; ".join(warnings) if warnings else None,
        "has_images": has_images,
    }


def extract_tables_for_page(page: "fitz.Page", plumber_page) -> dict[str, Any]:
    rendered: list[str] = []
    count = 0

    if plumber_page is not None:
        try:
            tables = plumber_page.extract_tables() or []
        except Exception as exc:
            logger.warning("pdfplumber tables failed: %s", exc)
            tables = []
        for table in tables:
            if not table:
                continue
            rows = []
            for row in table:
                cells = ["" if c is None else str(c).replace("\n", " ").strip() for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                count += 1
                rendered.append("\n".join(rows))

    if count == 0:
        try:
            found = page.find_tables()
            tables = found.tables if found else []
        except Exception:
            tables = []
        for tab in tables:
            try:
                data = tab.extract()
            except Exception:
                continue
            rows = []
            for row in data or []:
                cells = ["" if c is None else str(c).replace("\n", " ").strip() for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                count += 1
                rendered.append("\n".join(rows))

    text = "\n\n".join(rendered).strip() or None
    return {"table_count": count, "table_text": text}


# ---------------------------------------------------------------------------
# Cleaning (must not destroy clinical numbers)
# ---------------------------------------------------------------------------

_PROTECTED_LINE = re.compile(
    rf"({GE}|{LE}|[%]|cm|mm|years?|\bClass\s+I|\bGrade\s+[ABCDI]|\bLevel\s+[ABC]|"
    rf"\b\d+\.\d+\.\d+\b|\b(do not offer|offer|consider)\b)",
    re.I,
)


def merge_wrapped_lines(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return text
    out = [lines[0]]
    for line in lines[1:]:
        prev = out[-1].rstrip()
        curr = line.strip()
        if not prev or not curr:
            out.append(line)
            continue
        if prev.lstrip().startswith("|") or curr.startswith("|"):
            out.append(line)
            continue
        if curr.startswith(("#", "-", "*", BULLET)) or re.match(r"^\d+[.)]\s", curr):
            out.append(line)
            continue
        if prev.endswith(":"):
            out.append(line)
            continue
        if curr[:1].islower():
            out[-1] = prev + " " + curr
        else:
            out.append(line)
    return "\n".join(out)


def clean_text(text: str, headers: list[str] | None = None, footers: list[str] | None = None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\u00ad", "")
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    # Encoding repair only: replacement char between digits is almost always an en-dash.
    text = re.sub(rf"(?<=\d){REPLACEMENT}(?=\d)", EN_DASH, text)
    # Join syllabic hyphenation only (letter-letter). Never join digit ranges: 0.98-\n1.00
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "", text)
    text = merge_wrapped_lines(text)

    header_set = {h.strip() for h in (headers or []) if h.strip()}
    footer_set = {f.strip() for f in (footers or []) if f.strip()}
    raw_lines = text.split("\n")
    n = len(raw_lines)
    kept: list[str] = []
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        edge_header = stripped in header_set and i < 4
        edge_footer = stripped in footer_set and i >= max(0, n - 4)
        if (edge_header or edge_footer) and not _PROTECTED_LINE.search(stripped):
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_repeated_headers_footers(page_texts: list[str]) -> tuple[list[str], list[str]]:
    if len(page_texts) < 3:
        return [], []
    min_count = max(3, int(len(page_texts) * 0.5))

    def edge(text: str, which: str) -> list[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        sample = lines[:2] if which == "header" else lines[-2:]
        out = []
        for ln in sample:
            if len(ln) > 100 or _PROTECTED_LINE.search(ln):
                continue
            out.append(ln)
        return out

    hc: Counter[str] = Counter()
    fc: Counter[str] = Counter()
    for text in page_texts:
        hc.update(edge(text, "header"))
        fc.update(edge(text, "footer"))
    headers = [k for k, c in hc.items() if c >= min_count]
    footers = [k for k, c in fc.items() if c >= min_count]
    return headers, footers


def hyphenation_unit_tests() -> list[dict[str, Any]]:
    cases = [
        {
            "name": "letter hyphenation joined",
            "raw": "coun-\ntries",
            "expect_contains": "countries",
            "forbid": "coun-tries",
        },
        {
            "name": "decimal range preserved",
            "raw": "0.98-\n1.00",
            "expect_contains": "0.98-1.00",
            "forbid": "0.981.00",
        },
        {
            "name": "en-dash range preserved",
            "raw": f"0.98{EN_DASH}1.00",
            "expect_contains": f"0.98{EN_DASH}1.00",
            "forbid": "0.981.00",
        },
        {
            "name": "page range preserved",
            "raw": "301-304",
            "expect_contains": "301-304",
            "forbid": "301304",
        },
        {
            "name": "measurement preserved",
            "raw": "5.5 cm",
            "expect_contains": "5.5 cm",
            "forbid": "55 cm",
        },
        {
            "name": "age range preserved",
            "raw": "65-75 years",
            "expect_contains": "65-75 years",
            "forbid": "6575",
        },
    ]
    results = []
    for case in cases:
        cleaned = clean_text(case["raw"])
        folded = cleaned.replace("\n", "")
        ok = case["expect_contains"] in folded and case["forbid"] not in folded.replace(" ", "")
        results.append({**case, "clean": cleaned, "pass": ok})
    return results


# ---------------------------------------------------------------------------
# Numeric preservation
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    rf"""
    (?:[{GE}{LE}<>~\u2248]\s*)?
    \d+(?:[.,]\d+)?
    (?:\s*[{EN_DASH}{MINUS_SIGN}\-]\s*\d+(?:[.,]\d+)?)?
    (?:\s*(?:%|cm|mm|kg|years?|yrs?|months?|weeks?|days?|hours?))?
    """,
    re.I | re.X,
)

_CRITICAL_RE = re.compile(
    rf"[{GE}{LE}%]|cm|mm|years?|yrs?|months?|weeks?",
    re.I,
)


def fold_numeric_text(text: str) -> str:
    text = (text or "").replace("\n", " ")
    text = text.replace(EN_DASH, "-").replace(MINUS_SIGN, "-")
    text = re.sub(r"\s+", " ", text)
    return text


def numeric_tokens(text: str) -> list[str]:
    folded = fold_numeric_text(text)
    return [m.group(0).strip() for m in _NUMERIC_RE.finditer(folded)]


def _norm_token(tok: str) -> str:
    t = fold_numeric_text(tok).lower().replace(" ", "")
    t = t.replace(GE, ">=").replace(LE, "<=")
    return t


def is_critical_token(tok: str) -> bool:
    return bool(_CRITICAL_RE.search(tok)) or bool(re.search(r"\d+[.,]\d+", tok)) or bool(
        re.search(rf"\d+\s*[{EN_DASH}\-]\s*\d+", tok)
    )


def compare_numeric(raw_text: str, clean_text_value: str, header_footer_text: str = "") -> dict[str, Any]:
    raw_toks = numeric_tokens(raw_text)
    clean_toks = numeric_tokens(clean_text_value)
    raw_bag = Counter(_norm_token(t) for t in raw_toks)
    clean_bag = Counter(_norm_token(t) for t in clean_toks)
    missing_norm = list((raw_bag - clean_bag).elements())
    hf_bag = Counter(_norm_token(t) for t in numeric_tokens(header_footer_text))
    examples = []
    remaining = Counter(missing_norm)
    for tok in raw_toks:
        n = _norm_token(tok)
        if remaining[n]:
            examples.append(tok)
            remaining[n] -= 1
    noncritical = []
    critical = []
    for tok in examples:
        n = _norm_token(tok)
        if hf_bag[n] and not is_critical_token(tok):
            noncritical.append(tok)
            hf_bag[n] -= 1
        elif is_critical_token(tok):
            critical.append(tok)
        else:
            noncritical.append(tok)
    n_raw = len(raw_toks)
    n_missing = len(examples)
    return {
        "n_numeric_raw": n_raw,
        "n_numeric_clean": len(clean_toks),
        "n_numeric_missing": n_missing,
        "numeric_loss_ratio": (n_missing / n_raw) if n_raw else 0.0,
        "missing_numeric_examples": examples[:20],
        "critical_numeric_loss": bool(critical),
        "critical_numeric_examples": critical[:20],
        "noncritical_missing_examples": noncritical[:20],
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_MD_SECTION = re.compile(r"^(#{1,6})\s+(.+)$")
_NUMBERED_SECTION = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][A-Za-z][\w ,/&:-]{1,90})$")
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,/&:-]{7,90}$")
_KNOWN_HEADINGS = {
    "screening",
    "diagnosis",
    "surveillance",
    "repair",
    "management",
    "risk factors",
    "treatment",
    "endovascular repair",
    "open surgical repair",
    "prevention of aaa",
    "prevention of aaa rupture",
    "identifying people with abdominal aortic aneurysms",
    "monitoring and reducing the risk of rupture",
    "reducing the risk of rupture",
    "assessing and reducing the risk of rupture",
    "imaging technique",
    "recommendation statement",
}

# Navigational / agenda headings that name a slide or contents page rather than a
# clinical section. They were previously treated as real sections, which made a
# single label (e.g. "Areas of focus") inherit across dozens of unrelated pages.
_NAVIGATIONAL_HEADINGS = {
    "areas of focus",
    "highlights",
    "contents",
    "table of contents",
    "overview",
    "outline",
    "agenda",
    "recommendation",
    "level of recommendation",
    "quality of evidence",
}

# A page may keep inheriting the last detected heading for at most this many
# pages. After that the section is reported as unknown rather than guessed.
MAX_SECTION_INHERITANCE_PAGES = 5

_HEADING_LINE_STOP = (".", ",", ";", ":", "?", "!")


def _looks_like_heading_line(line: str) -> bool:
    """True when a page's first line reads like a slide/section title.

    Deliberately conservative: short, no terminal punctuation, not a bullet or a
    numbered recommendation, and containing real words. This recovers the real
    SVS slide titles ("Aneurysm imaging", "The decision to treat") and NICE
    section headings ("Repairing unruptured aneurysms") that the previous rules
    missed.
    """
    line = (line or "").strip()
    if not (2 <= len(line) <= 70):
        return False
    if line.endswith(_HEADING_LINE_STOP):
        return False
    if re.match(r"^[•●•\-\*\(]", line):
        return False
    if re.match(r"^\d", line):
        return False
    words = line.split()
    if not (1 <= len(words) <= 10):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    if re.sub(r"\s+", " ", line).strip(" :").lower() in _NAVIGATIONAL_HEADINGS:
        return False
    # Reject mid-sentence fragments such as "renal insufficiency, and diabetes"
    if not line[0].isupper():
        return False
    return True


def headings_on_page(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    lines = [ln.strip() for ln in (text or "").splitlines()]
    first_nonempty = next((ln for ln in lines if ln), "")
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or len(line) > 120:
            continue
        folded_nav = re.sub(r"\s+", " ", line).strip(" :").lower()
        if folded_nav in _NAVIGATIONAL_HEADINGS:
            continue
        if re.match(r"^\d+\.\d+\.\d+", line):
            continue
        m = _MD_SECTION.match(line)
        if m:
            found.append(("h" + str(len(m.group(1))), m.group(2).strip()))
            continue
        m = _NUMBERED_SECTION.match(line)
        if m:
            title_body = m.group(2).strip()
            if title_body.endswith((",", ".", ";", ":")):
                continue
            if re.search(r"\b(way|street|avenue|commons|university|department|email)\b", title_body, re.I):
                continue
            if len(title_body.split()) > 12:
                continue
            nums = m.group(1)
            title = f"{nums} {title_body}"
            level = "section" if "." not in nums else "subsection"
            found.append((level, title))
            continue
        folded = re.sub(r"\s+", " ", line).strip(" :")
        if folded.lower() in _KNOWN_HEADINGS:
            found.append(("section", folded))
            continue
        letters = [c for c in line if c.isalpha()]
        if (
            letters
            and len(letters) >= 8
            and sum(c.isupper() for c in letters) / len(letters) >= 0.9
            and not line.endswith(".")
            and _ALLCAPS.match(line)
        ):
            found.append(("section", line))
    # Slide decks and NICE section pages put the real heading on the first line
    # without any of the markers the rules above look for.
    if not any(level in {"h1", "h2", "section"} for level, _ in found):
        if _looks_like_heading_line(first_nonempty):
            found.insert(0, ("section", re.sub(r"\s+", " ", first_nonempty).strip(" :")))
    return found


def update_section_state(text: str, state: dict[str, str | None]) -> dict[str, Any]:
    found = headings_on_page(text)
    if not found:
        inherited_for = int(state.get("pages_since_heading") or 0) + 1
        state["pages_since_heading"] = inherited_for
        if state.get("section_title") and inherited_for <= MAX_SECTION_INHERITANCE_PAGES:
            return {
                "section_title": state["section_title"],
                "section_source": "inherited",
            }
        # Stop guessing once the last detected heading is too far behind.
        state["section_title"] = None
        return {"section_title": None, "section_source": "unknown"}

    section = state.get("section_title")
    applied = False
    for level, title in found:
        if level in {"h1", "h2", "section"}:
            section = title
            applied = True
        elif section is None:
            section = title
            applied = True
    if applied:
        state["section_title"] = section
        state["pages_since_heading"] = 0
        return {"section_title": section, "section_source": "detected"}

    # Only subsections were seen; the active section carries over unchanged.
    inherited_for = int(state.get("pages_since_heading") or 0) + 1
    state["pages_since_heading"] = inherited_for
    if section and inherited_for <= MAX_SECTION_INHERITANCE_PAGES:
        return {"section_title": section, "section_source": "inherited"}
    state["section_title"] = None
    return {"section_title": None, "section_source": "unknown"}


# ---------------------------------------------------------------------------
# Recommendations (source-supported only)
# ---------------------------------------------------------------------------

_CLASS_RE = re.compile(r"\bClass\s+(IIa|IIb|III|II|I)\b", re.I)
_LEVEL_RE = re.compile(r"\bLevel\s+([ABC])\b", re.I)
_GRADE_RE = re.compile(r"\bGrade\s+([ABCDI])\b", re.I)
_NICE_ID_INLINE = re.compile(r"(?m)^([1-9]\.\d{1,2}\.\d{1,3})\s+(.+)$")
_DOI_LINE_RE = re.compile(r"\bdoi\b|10\.\d{4,}/", re.I)
_NICE_VERB_RE = re.compile(r"\b(do not offer|offer|consider)\b", re.I)
_SVS_WE_RE = re.compile(r"^[^\w]{0,4}We (recommend|suggest)\b", re.I)
_SVS_GLOSSARY_RE = re.compile(
    r"Benefits of an intervention outweighed|Benefits and risks are less certain|Strength of Recommendation",
    re.I,
)


def _uniq_join(values: list[str]) -> str | None:
    seen: list[str] = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            seen.append(v)
    return "; ".join(seen) if seen else None


def _confidence(has_id: bool, has_body: bool, structured_grade: bool) -> str:
    if has_id and has_body:
        return "high"
    if has_body and structured_grade:
        return "medium"
    if has_body or has_id:
        return "medium"
    return "low"


def extract_nice_recommendations(raw_text: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if not raw_text:
        return recs
    tokens = re.split(r"(?m)^([1-9]\.\d{1,2}\.\d{1,3})\s*$", raw_text)
    i = 1
    while i + 1 < len(tokens):
        rec_id = tokens[i].strip()
        body = tokens[i + 1]
        i += 2
        body = re.split(r"(?m)^(?:Page \d+|Abdominal aortic aneurysm: diagnosis|[1-9]\.\d+\s+[A-Z])", body)[0]
        body_clean = re.sub(r"[ \t]+", " ", body).strip()
        body_clean = re.sub(r"\n{2,}", "\n", body_clean)
        if len(body_clean) < 20 or _looks_like_toc(body_clean):
            continue
        if _DOI_LINE_RE.search(body_clean) and len(body_clean) < 80:
            continue
        verb = None
        vm = _NICE_VERB_RE.search(body_clean)
        if vm:
            verb = vm.group(1).lower()
        recs.append(
            {
                "recommendation_id": rec_id,
                "recommendation_text": body_clean,
                "recommendation_class": None,
                "recommendation_grade": None,
                "evidence_level": None,
                "recommendation_strength": verb,
                "source_excerpt": body_clean,
                "extraction_confidence": _confidence(True, True, False),
            }
        )
    if not recs:
        for m in _NICE_ID_INLINE.finditer(raw_text):
            rec_id, rest = m.group(1), m.group(2).strip()
            if len(rest) < 20:
                continue
            if _DOI_LINE_RE.search(rest):
                continue
            verb = None
            vm = _NICE_VERB_RE.search(rest)
            if vm:
                verb = vm.group(1).lower()
            recs.append(
                {
                    "recommendation_id": rec_id,
                    "recommendation_text": rest,
                    "recommendation_class": None,
                    "recommendation_grade": None,
                    "evidence_level": None,
                    "recommendation_strength": verb,
                    "source_excerpt": m.group(0).strip(),
                    "extraction_confidence": _confidence(True, True, False),
                }
            )
    return recs


def extract_uspstf_recommendations(raw_text: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if not raw_text:
        return recs
    sentences = re.split(r"(?<=[.!?])\s+", raw_text.replace("\n", " "))
    for sent in sentences:
        s = sent.strip()
        if not re.search(r"\bUSPSTF recommends\b", s, re.I):
            continue
        grade = None
        gm = re.search(r"\(([ABCDI]) recommendation\)|\b([ABCDI]) recommendation\b|\b(I statement)\b", s, re.I)
        if gm:
            if gm.group(3):
                grade = "I statement"
            else:
                letter = (gm.group(1) or gm.group(2)).upper()
                grade = f"{letter} recommendation"
        recs.append(
            {
                "recommendation_id": None,
                "recommendation_text": s,
                "recommendation_class": None,
                "recommendation_grade": grade,
                "evidence_level": None,
                "recommendation_strength": None,
                "source_excerpt": s,
                "extraction_confidence": "high" if grade else "medium",
            }
        )
    return recs


def extract_svs_recommendations(raw_text: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if not raw_text:
        return recs
    if _SVS_GLOSSARY_RE.search(raw_text) and raw_text.lower().count("we recommend") <= 2:
        # GRADE definition slide, not a clinical recommendation list.
        if re.search(r'Strength of Recommendation|Level of Evidence', raw_text):
            return recs
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.fullmatch(r'["“]?We recommend["”]?', line, re.I) or re.fullmatch(r'["“]?We suggest["”]?', line, re.I):
            i += 1
            continue
        in_text = re.search(r"\bwe (recommend|suggest)\b", line, re.I)
        if in_text and not _SVS_GLOSSARY_RE.search(" ".join(lines[i : i + 4])):
            buf = [lines[i].strip()]
            j = i + 1
            strength = None
            evidence = None
            while j < len(lines):
                nxt = lines[j].strip()
                if re.search(r"\bwe (recommend|suggest)\b", nxt, re.I) and len(nxt) > 20:
                    break
                if re.fullmatch(r"[12]", nxt) and strength is None:
                    strength = nxt
                    if j + 1 < len(lines) and re.fullmatch(r"[ABC]", lines[j + 1].strip()):
                        evidence = lines[j + 1].strip()
                        j += 2
                        break
                    j += 1
                    break
                if re.fullmatch(r"[ABC]", nxt) and strength is not None:
                    evidence = nxt
                    j += 1
                    break
                if nxt:
                    buf.append(nxt)
                j += 1
            body = " ".join(buf).strip()
            if len(body) >= 30:
                recs.append(
                    {
                        "recommendation_id": None,
                        "recommendation_text": body,
                        "recommendation_class": None,
                        "recommendation_grade": None,
                        "evidence_level": evidence,
                        "recommendation_strength": strength,
                        "source_excerpt": body,
                        "extraction_confidence": "medium",
                    }
                )
            i = j
            continue
        i += 1
    return recs


def extract_generic_labeled_recommendations(raw_text: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for sent in re.split(r"(?<=[.!?])\s+", (raw_text or "").replace("\n", " ")):
        s = sent.strip()
        if re.match(r"(?i)^(recommendation|we recommend)\b", s) and len(s) > 40:
            recs.append(
                {
                    "recommendation_id": None,
                    "recommendation_text": s,
                    "recommendation_class": None,
                    "recommendation_grade": None,
                    "evidence_level": None,
                    "recommendation_strength": None,
                    "source_excerpt": s,
                    "extraction_confidence": "low",
                }
            )
    return recs


def extract_recommendations_for_document(raw_text: str, document_id: str, page_status: str) -> list[dict[str, Any]]:
    if page_status in {"low_text", "image_only", "ocr_required", "corrupted"}:
        return []
    if document_id == DOC_ID_NICE:
        return extract_nice_recommendations(raw_text)
    if document_id == DOC_ID_USPSTF:
        return extract_uspstf_recommendations(raw_text)
    if document_id == DOC_ID_SVS:
        return extract_svs_recommendations(raw_text)
    return extract_generic_labeled_recommendations(raw_text)


# ---------------------------------------------------------------------------
# Document-level metadata (source-supported only)
# ---------------------------------------------------------------------------


def _first(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = re.search(pat, text or "", re.I)
        if m:
            return m.group(0)
    return None


def _pdf_year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"(20\d{2})", value)
    return int(m.group(1)) if m else None


def extract_document_level_fields(pdf_path: Path, identity: dict[str, Any], page1: str, sample_text: str) -> dict[str, Any]:
    meta = pdf_metadata(pdf_path)
    pdf_title = _clean_pdf_title(meta.get("title"))
    author = (meta.get("author") or "").strip() or None
    if author and author.lower() in {"(unspecified)", "unknown"}:
        author = None

    blob = f"{page1}\n{sample_text[:4000]}"
    organization = None
    org_source = None
    nice_org = _first(blob, [r"National Institute for Health and Care Excellence(?: \(NICE\))?"])
    uspstf_org = _first(blob, [r"U\.?S\.?\s+Preventive Services Task Force"])
    # Checked before svs_org: the ESVS name contains the SVS name as a substring.
    esvs_org = _first(blob, [r"European Society for Vascular Surgery(?: \(ESVS\))?"])
    svs_org = _first(blob, [r"Society for Vascular Surgery"])
    if nice_org:
        organization, org_source = nice_org, "page text"
    elif uspstf_org:
        organization, org_source = uspstf_org, "page text"
    elif esvs_org:
        organization, org_source = esvs_org, "page text"
    elif svs_org:
        organization, org_source = svs_org, "page text"
    elif author and re.search(
        r"National Institute for Health and Care Excellence|Preventive Services Task Force|Society for Vascular Surgery",
        author,
        re.I,
    ):
        organization, org_source = author, "PDF Author metadata"

    publication_year = None
    year_source = None
    year_patterns = [
        (r"Published:\s+\d{1,2}\s+\w+\s+(20\d{2})", "Published date in text"),
        (r"Accepted for Publication:\s+\w+\s+\d{1,2},\s+(20\d{2})", "Accepted for Publication date in text"),
        (r"JAMA\.\s+(20\d{2});", "JAMA citation year in text"),
        (r"\b(20\d{2}) Practice Guidelines", "Practice Guidelines year in text"),
        (r"\(ESVS\)\s+(20\d{2})\s+Clinical", "ESVS guideline year in title"),
        (r"Eur J Vasc Endovasc Surg\s*\((20\d{2})\)", "journal citation year in text"),
        (r"\[published\s+\w+\s+\d{1,2},\s+(20\d{2})\]", "published date in text"),
    ]
    for pat, label in year_patterns:
        m = re.search(pat, blob, re.I)
        if m:
            publication_year = int(m.group(1))
            year_source = label
            break

    source_url = None
    url_hit = _first(
        blob,
        [
            r"https?://www\.nice\.org\.uk/guidance/ng156",
            r"www\.nice\.org\.uk/guidance/ng156",
            r"doi:10\.1001/jama\.\d{4}\.\d+",
            r"vsweb\.org/Guidelines",
            r"https://doi\.org/10\.1016/j\.ejvs\.\d{4}\.\d+\.\d+",
        ],
    )
    if url_hit:
        source_url = url_hit

    authors: list[str] | None = None
    if author and not re.search(r"literature review|nathan haas", author, re.I):
        if organization and author.strip().lower() == organization.strip().lower():
            authors = None
        else:
            authors = [author]

    document_type = identity["document_type"]
    if identity["document_id"] == DOC_ID_NICE:
        document_type = "official guideline"
        source_type = "official guideline"
    elif identity["document_id"] == DOC_ID_USPSTF:
        document_type = "government/public health recommendation"
        source_type = "government/public health recommendation"
    elif identity["document_id"] == DOC_ID_SVS:
        document_type = "official guideline"
        source_type = "official guideline"
    elif identity["document_id"] == DOC_ID_ESVS:
        document_type = "official guideline"
        source_type = "official guideline"
    elif document_type == "review article" or re.search(r"literature review", blob + " " + (author or ""), re.I):
        document_type = "review article"
        source_type = "review article"
    else:
        source_type = "other"

    credibility_parts = []
    if organization:
        credibility_parts.append(f"Organization stated in {org_source}: {organization}.")
    if publication_year:
        credibility_parts.append(f"Publication year from {year_source}: {publication_year}.")
    if source_url:
        credibility_parts.append(f"Source locator printed in the document: {source_url}.")
    if identity["document_id"] == DOC_ID_SVS:
        credibility_parts.append("Title page identifies 2018 Practice Guidelines from the Society for Vascular Surgery.")
        if author and re.search(r"nathan haas", author, re.I):
            credibility_parts.append("PDF Author metadata (Nathan Haas) was not used as guideline authorship.")
    if identity["document_id"] == DOC_ID_USPSTF:
        jama = _first(blob, [r"JAMA\.\s+2019;322\(22\):2211-2218"])
        if jama:
            credibility_parts.append(f"Bibliographic citation present in the PDF: {jama}.")
    if document_type == "review article":
        credibility_parts.append("Document presents itself as a literature review; no journal name or DOI was found in the file.")
    if identity["document_id"] == DOC_ID_ESVS:
        cc = _first(blob, [r"creativecommons\.org/licenses/by/4\.0"])
        if cc:
            credibility_parts.append(
                f"Open-access licence printed in the document: {cc} (CC BY 4.0)."
            )
    credibility_note = " ".join(credibility_parts) if credibility_parts else None

    public_access = None
    if source_url and re.search(r"nice\.org\.uk|vsweb\.org", source_url, re.I):
        public_access = True
    if identity["document_id"] == DOC_ID_ESVS and _first(blob, [r"creativecommons\.org/licenses/by/4\.0"]):
        public_access = True

    document_name = identity["document_name"]
    if identity["document_id"] == DOC_ID_SVS:
        document_name = "Care of Patients with an Abdominal Aortic Aneurysm"
    elif pdf_title:
        document_name = pdf_title

    return {
        "document_name": document_name,
        "document_type": document_type,
        "source_type": source_type,
        "source_organization": organization,
        "authors": authors,
        "publication_year": publication_year,
        "source_url": source_url,
        "credibility_note": credibility_note,
        "public_access": public_access,
        "year_source": year_source,
        "identification_method": identity.get("identification_method"),
    }


# ---------------------------------------------------------------------------
# Process one PDF / all PDFs
# ---------------------------------------------------------------------------


def process_pdf(pdf_path: Path, project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    identity = identify_document(pdf_path, project_root)
    warnings: list[str] = []
    page_records: list[dict[str, Any]] = []
    rec_records: list[dict[str, Any]] = []

    plumber_doc = None
    if HAS_PDFPLUMBER:
        try:
            plumber_doc = pdfplumber.open(pdf_path)
        except Exception as exc:
            warnings.append(f"pdfplumber open failed: {exc}")
            plumber_doc = None

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        if plumber_doc is not None:
            plumber_doc.close()
        raise RuntimeError(f"PyMuPDF failed to open {pdf_path.name}: {exc}") from exc

    try:
        n_pages = doc.page_count
        extracted: list[dict[str, Any]] = []
        for i in range(n_pages):
            page = doc[i]
            ext = extract_page_with_fallback(page)
            plumber_page = plumber_doc.pages[i] if plumber_doc is not None and i < len(plumber_doc.pages) else None
            if (not ext["text"].strip()) and plumber_page is not None:
                fb = plumber_page.extract_text() or ""
                if fb.strip():
                    ext["text"] = fb
                    ext["extraction_library"] = "pdfplumber"
                    ext["extraction_status"] = classify_extraction_status(fb, ext.get("has_images", False))
                    ext["extraction_warning"] = "used pdfplumber text fallback"
            tables = extract_tables_for_page(page, plumber_page)
            extracted.append({**ext, **tables})

        raw_pages = [e["text"] for e in extracted]
        headers, footers = detect_repeated_headers_footers(raw_pages)
        hf_text = "\n".join(headers + footers)
        page1 = raw_pages[0] if raw_pages else ""
        doc_fields = extract_document_level_fields(pdf_path, identity, page1, "\n".join(raw_pages[:4]))
        identity["document_name"] = doc_fields["document_name"]
        identity["document_type"] = doc_fields["document_type"]
        section_state: dict[str, str | None] = {}
        libraries = set()

        for i, ext in enumerate(extracted):
            page_number = i + 1
            raw = ext["text"]
            cleaned = clean_text(raw, headers=headers, footers=footers)
            numeric = compare_numeric(raw, cleaned, header_footer_text=hf_text)
            if numeric["critical_numeric_loss"]:
                warnings.append(
                    f"page {page_number} critical numeric loss: {numeric['critical_numeric_examples']}"
                )
            sec = update_section_state(cleaned or raw, section_state)
            recs = extract_recommendations_for_document(
                raw, identity["document_id"], ext["extraction_status"]
            )
            libraries.add(ext["extraction_library"])
            working = cleaned or raw
            record = {
                "document_id": identity["document_id"],
                "document_name": identity["document_name"],
                "document_type": identity["document_type"],
                "source_file": identity["source_file"],
                "page_number": page_number,
                "section_title": sec["section_title"],
                "section_source": sec["section_source"],
                "raw_text": raw,
                "clean_text": cleaned,
                "table_text": ext.get("table_text"),
                "character_count": len(working),
                "word_count": len(re.findall(r"\S+", working)),
                "line_count": len(working.splitlines()) if working else 0,
                "extraction_status": ext["extraction_status"],
                "extraction_library": ext["extraction_library"],
            }
            page_records.append(record)

            for rec in recs:
                rec_records.append(
                    {
                        "recommendation_id": rec.get("recommendation_id"),
                        "document_id": identity["document_id"],
                        "document_name": identity["document_name"],
                        "document_type": identity["document_type"],
                        "page_number": page_number,
                        "section_title": sec["section_title"],
                        "recommendation_text": rec.get("recommendation_text"),
                        "source_excerpt": rec.get("source_excerpt"),
                        "recommendation_grade": rec.get("recommendation_grade"),
                        "recommendation_class": rec.get("recommendation_class"),
                        "evidence_level": rec.get("evidence_level"),
                        "extraction_confidence": rec.get("extraction_confidence"),
                    }
                )
    finally:
        doc.close()
        if plumber_doc is not None:
            plumber_doc.close()

    page_statuses = [p["extraction_status"] for p in page_records]
    if any(s == "corrupted" for s in page_statuses):
        doc_status = "corrupted"
    elif any(s == "ocr_required" for s in page_statuses):
        doc_status = "ocr_required"
    elif any(s in {"low_text", "image_only"} for s in page_statuses):
        doc_status = "partial"
    else:
        doc_status = "ok"

    extraction_library = "+".join(sorted(libraries)) if libraries else "pymupdf"
    doc_meta = {
        "document_id": identity["document_id"],
        "document_name": identity["document_name"],
        "document_type": identity["document_type"],
        "source_file": identity["source_file"],
        "source_path": relative_posix(pdf_path, project_root),
        "source_organization": doc_fields.get("source_organization"),
        "authors": doc_fields.get("authors"),
        "publication_year": doc_fields.get("publication_year"),
        "source_type": doc_fields.get("source_type"),
        "credibility_note": doc_fields.get("credibility_note"),
        "public_access": doc_fields.get("public_access"),
        "source_url": doc_fields.get("source_url"),
        "page_count": len(page_records),
        "extraction_status": doc_status,
        "extraction_library": extraction_library,
        "_warnings": warnings,
        "_expected_pages": n_pages,
        "_identification_method": identity.get("identification_method"),
    }
    return page_records, rec_records, doc_meta


def process_all(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or find_project_root()
    pdf_files = discover_pdfs(project_root)
    pages: list[dict[str, Any]] = []
    recs: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    pipeline_warnings: list[str] = []

    hashes: dict[str, list[str]] = {}
    valid_pdfs: list[Path] = []
    for pdf in pdf_files:
        digest = file_md5(pdf)
        hashes.setdefault(digest, []).append(pdf.name)
        assessment = assess_pdf_validity(pdf)
        if not assessment["valid"]:
            excluded.append({"source_file": pdf.name, "reason": assessment["reason"]})
            continue
        valid_pdfs.append(pdf)

    for digest, names in hashes.items():
        if len(names) > 1:
            keep = names[0]
            for dup in names[1:]:
                if not any(e["source_file"] == dup for e in excluded):
                    excluded.append(
                        {
                            "source_file": dup,
                            "reason": f"Duplicate of {keep} (md5 {digest})",
                        }
                    )
            valid_pdfs = [p for p in valid_pdfs if p.name == keep or p.name not in names[1:]]

    for pdf in valid_pdfs:
        try:
            p_rows, r_rows, d_meta = process_pdf(pdf, project_root)
            pages.extend(p_rows)
            recs.extend(r_rows)
            docs.append(d_meta)
            logger.info("OK %s (%s pages)", pdf.name, d_meta["page_count"])
        except Exception as exc:
            logger.exception("FAILED %s", pdf.name)
            failed.append({"source_file": pdf.name, "error": str(exc)})
            pipeline_warnings.append(f"FAILED {pdf.name}: {exc}")

    pages_df = pd.DataFrame(pages)
    recs_df = pd.DataFrame(recs)
    docs_df = pd.DataFrame(docs)
    if not pages_df.empty:
        pages_df = pages_df.sort_values(["document_id", "page_number"]).reset_index(drop=True)
    return {
        "project_root": project_root,
        "pdf_files": pdf_files,
        "valid_pdfs": valid_pdfs,
        "pages_df": pages_df,
        "recommendations_df": recs_df,
        "documents_df": docs_df,
        "failed": failed,
        "excluded": excluded,
        "pipeline_warnings": pipeline_warnings,
    }


def _fold_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def excerpt_in_page(excerpt: str, raw_text: str, clean_text_value: str) -> bool:
    e = _fold_ws(excerpt)
    if len(e) < 8:
        return False
    return e in _fold_ws(raw_text) or e in _fold_ws(clean_text_value)


def owned_generated_files(processed_dir: Path) -> list[Path]:
    if not processed_dir.is_dir():
        return []
    owned: list[Path] = []
    for p in processed_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in {".csv", ".json", ".parquet"} or " " in p.name:
            owned.append(p)
    return owned


def cleanup_generated_outputs(
    project_root: Path,
    keep: set[str] | None = None,
    *,
    processed_only: bool = True,
) -> list[str]:
    keep = keep or set()
    deleted: list[str] = []
    processed = project_root / "data" / "processed"
    for path in owned_generated_files(processed):
        if path.name in keep:
            continue
        path.unlink()
        deleted.append(relative_posix(path, project_root))
    if not processed_only:
        for extra_dir_name in ("chunks", "embeddings"):
            extra_dir = project_root / "data" / extra_dir_name
            if extra_dir.is_dir():
                for path in extra_dir.rglob("*"):
                    if path.is_file() and path.name not in keep:
                        if path.suffix.lower() == ".pdf":
                            continue
                        path.unlink()
                        deleted.append(relative_posix(path, project_root))
    notebooks = project_root / "notebooks"
    if notebooks.is_dir():
        for path in notebooks.glob("*"):
            if path.is_file() and (" " in path.name) and path.suffix in {".json", ".ipynb", ".py"}:
                path.unlink()
                deleted.append(relative_posix(path, project_root))
        write_nb = notebooks / "_write_nb.py"
        if write_nb.exists():
            write_nb.unlink()
            deleted.append(relative_posix(write_nb, project_root))
    return deleted


def validate_dataset(result: dict[str, Any]) -> dict[str, Any]:
    pages_df: pd.DataFrame = result["pages_df"]
    recs_df: pd.DataFrame = result["recommendations_df"]
    docs_df: pd.DataFrame = result["documents_df"]
    errors: list[str] = []
    warnings: list[str] = list(result.get("pipeline_warnings") or [])

    if pages_df.empty:
        errors.append("No pages were extracted.")
        return {
            "overall": "FAIL",
            "errors": errors,
            "warnings": warnings,
            "numeric_tokens_raw": 0,
            "numeric_tokens_clean": 0,
            "numeric_tokens_missing": 0,
            "numeric_loss_ratio": 0.0,
            "critical_numeric_losses": [],
            "recommendation_traceability": "FAIL",
        }

    required = set(PAGE_COLUMNS)
    missing_cols = sorted(required - set(pages_df.columns))
    if missing_cols:
        errors.append("pages missing columns: " + ", ".join(missing_cols))

    if pages_df["raw_text"].fillna("").str.strip().eq("").all():
        errors.append("All raw_text values are empty.")

    n_raw = int(pages_df.apply(lambda r: len(numeric_tokens(r.get("raw_text") or "")), axis=1).sum())
    n_clean = int(pages_df.apply(lambda r: len(numeric_tokens(r.get("clean_text") or "")), axis=1).sum())
    critical_losses: list[dict[str, Any]] = []
    missing_total = 0
    for _, row in pages_df.iterrows():
        cmp = compare_numeric(row.get("raw_text") or "", row.get("clean_text") or "")
        missing_total += int(cmp["n_numeric_missing"])
        if cmp["critical_numeric_loss"]:
            critical_losses.append(
                {
                    "document_id": row["document_id"],
                    "page_number": int(row["page_number"]),
                    "examples": cmp["critical_numeric_examples"],
                }
            )
    if critical_losses:
        errors.append(f"Critical numeric losses on {len(critical_losses)} page(s).")

    glued = pages_df["clean_text"].fillna("").str.contains(r"0\.981\.00", regex=True)
    if glued.any():
        errors.append("Hyphen+newline bug still present (0.981.00) in clean_text.")

    rec_trace = "PASS"
    missing_excerpt = 0
    if recs_df is not None and not recs_df.empty:
        page_lookup = {
            (r["document_id"], int(r["page_number"])): r for _, r in pages_df.iterrows()
        }
        for _, rec in recs_df.iterrows():
            excerpt = rec.get("source_excerpt") or rec.get("recommendation_text")
            if not excerpt:
                missing_excerpt += 1
                continue
            key = (rec["document_id"], int(rec["page_number"]))
            page = page_lookup.get(key)
            if page is None or not excerpt_in_page(str(excerpt), page.get("raw_text") or "", page.get("clean_text") or ""):
                missing_excerpt += 1
        if missing_excerpt:
            rec_trace = "FAIL"
            errors.append(f"{missing_excerpt} recommendation(s) missing page traceability.")
        fake_ids = recs_df["recommendation_id"].dropna().astype(str).str.match(r"^(USPSTF_REC_|SVS_REC_|FAKE_)")
        if fake_ids.any():
            errors.append("Fabricated recommendation IDs were generated.")
            rec_trace = "FAIL"
        nice_on_other = recs_df[
            recs_df["recommendation_id"].fillna("").astype(str).str.match(r"^\d+\.\d+\.\d+$")
            & (recs_df["document_id"] != DOC_ID_NICE)
        ]
        if not nice_on_other.empty:
            errors.append("NICE-style recommendation IDs assigned to non-NICE documents.")
            rec_trace = "FAIL"
    else:
        warnings.append("No recommendations extracted.")

    for _, doc in docs_df.iterrows():
        expected = doc.get("_expected_pages")
        actual = doc.get("page_count")
        if expected is not None and actual is not None and int(expected) != int(actual):
            errors.append(f"{doc['document_id']} page count mismatch: expected {expected}, got {actual}")

    uspstf = docs_df[docs_df["document_id"] == DOC_ID_USPSTF]
    if not uspstf.empty:
        year = uspstf.iloc[0].get("publication_year")
        if year is not None and not (pd.isna(year)) and int(year) == 2014:
            errors.append("USPSTF publication_year incorrectly set to 2014 (prior-recommendation year).")

    overall = "FAIL" if errors else "PASS"
    return {
        "overall": overall,
        "errors": errors,
        "warnings": warnings,
        "numeric_tokens_raw": n_raw,
        "numeric_tokens_clean": n_clean,
        "numeric_tokens_missing": missing_total,
        "numeric_loss_ratio": (missing_total / n_raw) if n_raw else 0.0,
        "critical_numeric_losses": critical_losses,
        "recommendation_traceability": rec_trace,
        "output_validation": "PENDING",
    }


def build_extraction_report(result: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    pages_df: pd.DataFrame = result["pages_df"]
    recs_df: pd.DataFrame = result["recommendations_df"]
    status = pages_df["extraction_status"] if not pages_df.empty else pd.Series(dtype=str)
    rec_n = int(len(recs_df)) if recs_df is not None else 0
    hyphen_tests = hyphenation_unit_tests()
    return {
        "status": validation["overall"],
        "processing_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "documents": {
            "discovered": len(result["pdf_files"]),
            "processed": len(result["documents_df"]),
            "failed": len(result["failed"]),
            "excluded": len(result["excluded"]),
            "failed_details": result["failed"],
            "excluded_details": result["excluded"],
        },
        "pages": {
            "total": int(len(pages_df)),
            "ok": int((status == "ok").sum()),
            "low_text": int((status == "low_text").sum()),
            "image_only": int((status == "image_only").sum()),
            "ocr_required": int((status == "ocr_required").sum()),
            "corrupted": int((status == "corrupted").sum()),
        },
        "numeric_preservation": {
            "raw_numeric_tokens": validation["numeric_tokens_raw"],
            "clean_numeric_tokens": validation["numeric_tokens_clean"],
            "missing_numeric_tokens": validation["numeric_tokens_missing"],
            "loss_ratio": validation["numeric_loss_ratio"],
            "critical_losses": validation["critical_numeric_losses"],
        },
        "recommendations": {
            "total": rec_n,
            "with_grades": int(recs_df["recommendation_grade"].notna().sum()) if rec_n else 0,
            "with_evidence_levels": int(recs_df["evidence_level"].notna().sum()) if rec_n else 0,
            "with_source_excerpts": int(recs_df["source_excerpt"].notna().sum()) if rec_n else 0,
            "missing_traceability": 0 if validation["recommendation_traceability"] == "PASS" else None,
            "traceability": validation["recommendation_traceability"],
        },
        "hyphenation_tests": hyphen_tests,
        "ocr_available": HAS_OCR,
        "errors": validation["errors"],
        "warnings": validation["warnings"],
        "output_validation": validation.get("output_validation"),
    }


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    cleaned = df.where(pd.notnull(df), None)
    records = cleaned.to_dict(orient="records")
    for rec in records:
        for key, value in list(rec.items()):
            if isinstance(value, float) and pd.isna(value):
                rec[key] = None
            elif key in {"publication_year", "page_count", "page_number"} and isinstance(value, float) and value.is_integer():
                rec[key] = int(value)
    return records


def save_processed_outputs(result: dict[str, Any], report: dict[str, Any]) -> list[str]:
    root: Path = result["project_root"]
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    pages_df: pd.DataFrame = result["pages_df"].copy()
    recs_df: pd.DataFrame = result["recommendations_df"].copy()
    docs_df: pd.DataFrame = result["documents_df"].copy()

    for col in PAGE_COLUMNS:
        if col not in pages_df.columns:
            pages_df[col] = None
    pages_df = pages_df[PAGE_COLUMNS]

    doc_cols = [
        "document_id",
        "document_name",
        "document_type",
        "source_file",
        "source_path",
        "source_organization",
        "authors",
        "publication_year",
        "source_type",
        "credibility_note",
        "public_access",
        "source_url",
        "page_count",
        "extraction_status",
        "extraction_library",
    ]
    docs_public = docs_df.drop(columns=[c for c in docs_df.columns if str(c).startswith("_")], errors="ignore")
    for c in doc_cols:
        if c not in docs_public.columns:
            docs_public[c] = None
    docs_public = docs_public[doc_cols]

    rec_cols = [
        "recommendation_id",
        "document_id",
        "document_name",
        "document_type",
        "page_number",
        "section_title",
        "recommendation_text",
        "source_excerpt",
        "recommendation_grade",
        "recommendation_class",
        "evidence_level",
        "extraction_confidence",
    ]
    if recs_df.empty:
        recs_df = pd.DataFrame(columns=rec_cols)
    else:
        for c in rec_cols:
            if c not in recs_df.columns:
                recs_df[c] = None
        recs_df = recs_df[rec_cols]

    def dump(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pages_df.to_parquet(out / "pages_df.parquet", index=False)
    dump(out / "pages.json", _json_records(pages_df))
    dump(out / "document_metadata.json", _json_records(docs_public))
    dump(out / "recommendations.json", _json_records(recs_df))
    dump(out / "extraction_report.json", report)
    return [str((out / n).relative_to(root)) for n in sorted(PROCESSED_OUTPUT_NAMES)]


def verify_processed_files(project_root: Path) -> tuple[bool, list[str]]:
    processed = project_root / "data" / "processed"
    names = {p.name for p in processed.iterdir() if p.is_file()} if processed.is_dir() else set()
    extra = sorted(names - PROCESSED_OUTPUT_NAMES)
    missing = sorted(PROCESSED_OUTPUT_NAMES - names)
    errors = []
    if extra:
        errors.append("unexpected files in data/processed: " + ", ".join(extra))
    if missing:
        errors.append("missing required files: " + ", ".join(missing))
    for name in PROCESSED_OUTPUT_NAMES:
        path = processed / name
        if path.exists() and path.stat().st_size == 0:
            errors.append(f"empty file: {name}")
    return (not errors), errors


def print_preprocessing_summary(report: dict[str, Any]) -> None:
    docs = report["documents"]
    pages = report["pages"]
    nums = report["numeric_preservation"]
    recs = report["recommendations"]
    print("DOCUMENTS")
    print(f"- discovered: {docs['discovered']}")
    print(f"- processed: {docs['processed']}")
    print(f"- failed: {docs['failed']}")
    print(f"- excluded: {docs['excluded']}")
    if docs["excluded_details"]:
        for item in docs["excluded_details"]:
            print(f"  - {item['source_file']}: {item['reason']}")
    print()
    print("PAGES")
    print(f"- total: {pages['total']}")
    print(f"- ok: {pages['ok']}")
    print(f"- low_text: {pages['low_text']}")
    print(f"- image_only: {pages['image_only']}")
    print(f"- OCR required: {pages['ocr_required']}")
    print(f"- corrupted: {pages['corrupted']}")
    print()
    print("NUMERIC PRESERVATION")
    print(f"- raw numeric tokens: {nums['raw_numeric_tokens']}")
    print(f"- clean numeric tokens: {nums['clean_numeric_tokens']}")
    print(f"- missing tokens: {nums['missing_numeric_tokens']}")
    print(f"- loss ratio: {nums['loss_ratio']}")
    print(f"- critical losses: {nums['critical_losses']}")
    print()
    print("RECOMMENDATIONS")
    print(f"- total: {recs['total']}")
    print(f"- with grades: {recs['with_grades']}")
    print(f"- with evidence levels: {recs['with_evidence_levels']}")
    print(f"- with source excerpts: {recs['with_source_excerpts']}")
    print(f"- traceability: {recs['traceability']}")
    print()
    print("HYPHENATION TESTS")
    for case in report["hyphenation_tests"]:
        print(f"- {case['name']}: {'PASS' if case['pass'] else 'FAIL'}")
    print()
    print(f"STATUS: {report['status']}")
    if report["errors"]:
        print("\nERRORS:")
        for e in report["errors"]:
            print(" -", e)
    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"][:30]:
            print(" -", w)


def run_pipeline(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or find_project_root()
    deleted_before = cleanup_generated_outputs(project_root, keep=set(), processed_only=False)
    result = process_all(project_root)
    validation = validate_dataset(result)
    report = build_extraction_report(result, validation)
    save_processed_outputs(result, report)
    deleted_after = cleanup_generated_outputs(project_root, keep=PROCESSED_OUTPUT_NAMES, processed_only=True)
    ok, output_errors = verify_processed_files(project_root)
    if not ok:
        validation["output_validation"] = "FAIL"
        validation["overall"] = "FAIL"
        validation["errors"].extend(output_errors)
        report["output_validation"] = "FAIL"
        report["status"] = "FAIL"
        report["errors"] = validation["errors"]
        (project_root / "data" / "processed" / "extraction_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        validation["output_validation"] = "PASS"
        report["output_validation"] = "PASS"
        if validation["overall"] == "PASS":
            report["status"] = "PASS"
        (project_root / "data" / "processed" / "extraction_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    result["validation"] = validation
    result["report"] = report
    result["deleted_files"] = deleted_before + deleted_after
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    outcome = run_pipeline()
    print_preprocessing_summary(outcome["report"])
