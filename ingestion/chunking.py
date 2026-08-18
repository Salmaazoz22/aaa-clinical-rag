# -*- coding: utf-8 -*-
"""Section-aware chunking for the AAA clinical RAG corpus."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.preprocess import (
    EN_DASH,
    find_project_root,
    numeric_tokens,
    relative_posix,
)

# ---------------------------------------------------------------------------
# Chunk sizing
#
# Chunk size is budgeted in TOKENS, measured with the tokenizer of the embedding
# model that will actually encode the text. Character limits are kept only as a
# coarse guard rail: characters are not a proxy for tokens, and sizing by
# characters is what previously allowed 75.9% of chunks to overflow the model's
# token window and be silently truncated at encode time.
# ---------------------------------------------------------------------------
DEFAULT_EMBED_MODEL = "abhinand/MedEmbed-base-v0.1"

# Weights are pinned by commit so a hub-side update cannot silently change the
# production index underneath a frozen evaluation.
MODEL_REVISIONS = {
    "abhinand/MedEmbed-base-v0.1": "7a90c50263f620dff743eb9794b89a42bfc5d765",
}

# Upper bound on the embedding model's input window (MedEmbed-base-v0.1 -> 512).
# Only a fallback cap: the real limit is read from the model's own
# sentence_bert_config.json in `model_token_limit`.
MODEL_MAX_TOKENS = 512
# Reserve room for [CLS]/[SEP] so an encoded chunk never reaches the hard limit.
SPECIAL_TOKEN_BUDGET = 2
# Configurable chunk budget, expressed in content tokens.
TARGET_TOKENS = 220
OVERLAP_TOKENS = 40
MIN_TOKENS = 16

TARGET_CHARS = 900
OVERLAP_CHARS = 150
MAX_CHARS = 2400
MIN_VALID_CHARS = 50
SHORT_WARN_CHARS = 120

GLUED_RANGE_RE = re.compile(r"(?<!\d)0\.981\.00(?!\d)")
CORRUPT_RE = re.compile(r"\ufffd{3,}")

_TOKENIZER_CACHE: dict[str, Any] = {}


def max_content_tokens(model_name: str = DEFAULT_EMBED_MODEL) -> int:
    """Tokens of chunk text that can be embedded without truncation."""
    return model_token_limit(model_name) - SPECIAL_TOKEN_BUDGET


def model_revision(model_name: str = DEFAULT_EMBED_MODEL) -> str | None:
    """Pinned commit for this model, or None when it is not pinned."""
    return MODEL_REVISIONS.get(model_name)


def load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Load the embedding model at its pinned revision.

    Single entry point so the chunker, the index builder and query encoding can
    never drift onto different weights.
    """
    from sentence_transformers import SentenceTransformer

    revision = model_revision(model_name)
    if revision:
        return SentenceTransformer(model_name, revision=revision)
    return SentenceTransformer(model_name)


def get_tokenizer(model_name: str = DEFAULT_EMBED_MODEL):
    """Return (and cache) the tokenizer actually used by the embedding model."""
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision(model_name)
        )
    return _TOKENIZER_CACHE[model_name]


_LIMIT_CACHE: dict[str, int] = {}


def model_token_limit(model_name: str = DEFAULT_EMBED_MODEL) -> int:
    """Effective token limit at which the embedding model truncates.

    This must be the sentence-transformers `max_seq_length` of the ACTIVE model,
    NOT `tokenizer.model_max_length`. The two can differ sharply -- for
    all-MiniLM-L6-v2 the window is 256 while the tokenizer reports the
    backbone's 512 positions, which would understate the truncation risk by 2x.
    Resolved from the model's own sentence_bert_config.json, falling back to the
    tokenizer value capped by MODEL_MAX_TOKENS.
    """
    if model_name in _LIMIT_CACHE:
        return _LIMIT_CACHE[model_name]

    limit: int | None = None
    try:  # cached locally after first download; no network needed on re-runs
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(
            model_name, "sentence_bert_config.json", revision=model_revision(model_name)
        )
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        value = cfg.get("max_seq_length")
        if isinstance(value, int) and 0 < value <= 100000:
            limit = int(value)
    except Exception:
        limit = None

    if limit is None:
        tok_limit = getattr(get_tokenizer(model_name), "model_max_length", None)
        if isinstance(tok_limit, int) and 0 < tok_limit <= 100000:
            limit = min(int(tok_limit), MODEL_MAX_TOKENS)
        else:
            limit = MODEL_MAX_TOKENS

    _LIMIT_CACHE[model_name] = limit
    return limit


def count_tokens(text: str, model_name: str = DEFAULT_EMBED_MODEL) -> int:
    """Token count including the special tokens the encoder will add."""
    if not (text or "").strip():
        return 0
    tok = get_tokenizer(model_name)
    return len(tok.encode(text, add_special_tokens=True))


def load_processed(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or find_project_root()
    processed = project_root / "data" / "processed"
    pages_path = processed / "pages_df.parquet"
    recs_path = processed / "recommendations.json"
    if not pages_path.exists():
        raise FileNotFoundError(f"Missing {pages_path}; run notebook 01 first.")
    pages_df = pd.read_parquet(pages_path)
    recs = json.loads(recs_path.read_text(encoding="utf-8")) if recs_path.exists() else []
    recs_df = pd.DataFrame(recs)
    return {"project_root": project_root, "pages_df": pages_df, "recommendations_df": recs_df}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_split_points(text: str) -> list[int]:
    """Candidate split offsets that do not break numeric tokens."""
    points = {0, len(text)}
    for m in re.finditer(r"\n\s*\n", text):
        points.add(m.end())
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9])", text):
        points.add(m.end())
    for m in re.finditer(r"\n", text):
        points.add(m.end())
    protected: list[tuple[int, int]] = []
    for m in re.finditer(
        rf"\d+(?:[.,]\d+)?(?:\s*[{EN_DASH}\-]\s*\d+(?:[.,]\d+)?)?(?:\s*(?:%|cm|mm|years?))?",
        text,
        re.I,
    ):
        protected.append((m.start(), m.end()))

    def allowed(pos: int) -> bool:
        return all(not (a < pos < b) for a, b in protected)

    return sorted(p for p in points if allowed(p))


def _hard_token_split(text: str, budget: int, overlap: int, model_name: str) -> list[str]:
    """Last-resort split for a span with no safe character boundary.

    Splits on token ids so the budget is respected even when the text contains
    no sentence, line or paragraph break (e.g. a long unbroken table row).
    """
    tok = get_tokenizer(model_name)
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, budget - overlap)
    pieces: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start : start + budget]
        if not window:
            break
        piece = tok.decode(window, skip_special_tokens=True).strip()
        if piece:
            pieces.append(piece)
        if start + budget >= len(ids):
            break
    return pieces


def split_text(
    text: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    model_name: str = DEFAULT_EMBED_MODEL,
    min_tokens: int = MIN_TOKENS,
) -> list[str]:
    """Split text into pieces that fit the embedding model's token window.

    Split positions are chosen from `_safe_split_points`, so numeric tokens such
    as `5.5 cm` or `65-75 years` are never cut in half. The token budget is
    enforced against the real tokenizer, not against a character count.
    """
    text = (text or "").strip()
    if not text:
        return []

    limit = max_content_tokens(model_name)
    budget = min(int(target_tokens), limit)
    overlap_tokens = max(0, min(int(overlap_tokens), budget - 1))

    if count_tokens(text) <= limit:
        return [text]

    tok = get_tokenizer(model_name)
    points = _safe_split_points(text)
    n = len(text)
    pieces: list[str] = []
    start = 0
    guard = 0
    last_end = 0

    while start < n:
        guard += 1
        if guard > 10000:  # defensive: never loop forever on pathological input
            break
        remaining = text[start:].strip()
        if not remaining:
            break
        if len(tok.encode(remaining, add_special_tokens=False)) <= budget:
            pieces.append(remaining)
            break

        # Largest safe split point whose span still fits the token budget.
        candidates = [p for p in points if p > start]
        end = None
        lo, hi = 0, len(candidates) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            span = text[start : candidates[mid]]
            if len(tok.encode(span, add_special_tokens=False)) <= budget:
                end = candidates[mid]
                lo = mid + 1
            else:
                hi = mid - 1

        if end is None or end <= start:
            # No usable boundary inside the budget: fall back to a token split.
            for piece in _hard_token_split(text[start:], budget, overlap_tokens, model_name):
                pieces.append(piece)
            break

        if end <= last_end:
            # The frontier did not advance -- every safe point inside the budget
            # is already covered. Emitting here would just repeat a shrinking
            # suffix of the previous piece, so skip past what is already done.
            start = last_end
            continue

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        last_end = end
        if end >= n:
            break

        # Step back by `overlap_tokens` worth of text, snapped to a safe point.
        # The overlap is capped at half the piece just emitted: when a safe split
        # lands early, a fixed overlap could otherwise consume most of a short
        # piece and make the next chunk a near-copy of it.
        tail_ids = tok.encode(text[start:end], add_special_tokens=False)
        effective_overlap = min(overlap_tokens, len(tail_ids) // 2)
        if effective_overlap:
            tail_text = tok.decode(tail_ids[-effective_overlap:], skip_special_tokens=True)
            back = max(0, len(tail_text))
            desired = max(start + 1, end - back)
            snapped = [p for p in points if start < p <= desired]
            next_start = snapped[-1] if snapped else desired
        else:
            next_start = end
        start = max(next_start, start + 1)

    # Fold a trailing sliver into its predecessor when it still fits.
    if len(pieces) >= 2:
        tail = pieces[-1]
        if count_tokens(tail) < min_tokens:
            merged = pieces[-2] + "\n" + tail
            if count_tokens(merged) <= limit:
                pieces = pieces[:-2] + [merged]
    return pieces


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _compare_norm(text: str) -> str:
    """Normalise for redundancy comparison: drop cell separators and case."""
    t = (text or "").replace("|", " ")
    t = re.sub(r"[^0-9a-zA-Z.%<>=\-]+", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def table_is_redundant(table: str, clean: str, threshold: float = 0.6) -> bool:
    """True when `table` mostly restates `clean`.

    The extractor emits SVS recommendation tables twice: once as slide text and
    once as pipe-delimited cells. A plain substring test misses this because the
    pipes and reordered column headers break exact containment, so the same
    recommendations were stored twice inside one chunk.
    """
    nt, nc = _compare_norm(table), _compare_norm(clean)
    if not nt:
        return True
    if not nc:
        return False
    if nt in nc:
        return True
    # Some extracted tables lose all spacing ("MostAAAsareasymptomatic..."), so
    # word-level comparison cannot see that they restate the body text.
    nt_tight, nc_tight = nt.replace(" ", ""), nc.replace(" ", "")
    if len(nt_tight) >= 40 and nt_tight in nc_tight:
        return True
    words = nt.split()
    if len(words) < 5:
        return nt in nc
    grams = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
    hits = sum(1 for g in grams if g in nc)
    return (hits / len(grams)) >= threshold


def _as_optional(value: Any) -> str | None:
    """Normalise pandas NaN / empty strings to a real None.

    Reading pages back from parquet turns missing section titles into float NaN,
    which JSON-serialises to the bare string "nan" and is truthy. Both forms
    defeat presence checks, so missing metadata is normalised to None here.
    """
    text = _as_text(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def page_working_text(row: pd.Series) -> str:
    parts = []
    clean = _as_text(row.get("clean_text")).strip()
    table = _as_text(row.get("table_text")).strip()
    if clean:
        parts.append(clean)
    if table and not table_is_redundant(table, clean):
        parts.append("Table:\n" + table)
    return "\n\n".join(parts).strip()


def recs_for_page(recs_df: pd.DataFrame, document_id: str, page_number: int) -> list[dict[str, Any]]:
    if recs_df is None or recs_df.empty:
        return []
    subset = recs_df[(recs_df["document_id"] == document_id) & (recs_df["page_number"] == page_number)]
    return subset.to_dict(orient="records")


def attach_recommendation_metadata(chunk_text: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
    matched = []
    folded_chunk = _norm(chunk_text)
    for rec in recs:
        excerpt = _norm(str(rec.get("source_excerpt") or rec.get("recommendation_text") or ""))
        if excerpt and excerpt[:80] in folded_chunk:
            matched.append(rec)
            continue
        rec_id = rec.get("recommendation_id")
        if rec_id and str(rec_id) in chunk_text:
            matched.append(rec)
    if not matched:
        return {
            "recommendation_id": None,
            "recommendation_grade": None,
            "evidence_level": None,
        }
    rec = matched[0]
    return {
        "recommendation_id": rec.get("recommendation_id"),
        "recommendation_grade": rec.get("recommendation_grade"),
        "evidence_level": rec.get("evidence_level"),
    }


GUIDELINE_DOC_TYPES = {
    "official guideline",
    "government/public health recommendation",
}

_REFERENCE_SECTION_RE = re.compile(
    r"^\s*(references|bibliography|acknowledg|disclosure|conflict of interest)", re.I
)
# Bibliography markers. The last three cover the numbered reference style used by
# ESVS 2024 ("1261 Robertson L. Optimising intervals ...", "Available at: ...",
# "[Accessed 12 October 2023]"), which the journal-citation patterns alone miss.
_CITATION_RE = re.compile(
    r"\bet al\.?"
    r"|doi:\s*\S+"
    r"|\b\d{4};\s*\d+\s*[\(:]"
    r"|\bAvailable at:"
    r"|\[Accessed\b"
    r"|(?:^|\s)\d{1,4}\s+[A-Z][a-z]+ [A-Z]{1,3}[,.]",
    re.I | re.M,
)
# Contents-page leaders. Two styles occur in this corpus: solid runs ("....."),
# and space-separated runs (". . . . ."), which ESVS 2024 uses throughout its
# table of contents.
_DOT_LEADER_RE = re.compile(r"\.{6,}|(?:\.[ \t]){6,}")
_BOILERPLATE_RE = re.compile(
    r"all rights reserved|subject to notice of rights|terms-and\s*conditions|"
    r"\(reprinted\)\s*jama|clinical review & education",
    re.I,
)
# Recommendation language. A chunk carrying any of this is clinical content and
# is never discarded, however short it is or however many table cells it holds.
_RECOMMENDATION_RE = re.compile(
    r"\bwe recommend\b|\bwe suggest\b|\brecommends?\b|\bshould be offered\b|"
    r"\boffer\b|\bconsider\b|\bdo not offer\b|\brecommendation\b\s*\d|"
    r"\b\d\.\d\.\d+\b|\b(grade|level)\s*[a-d1-2]\b",
    re.I,
)
_CLINICAL_SIGNAL_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm|mm)|\bultraso|\bscreen|\bsurveillance\b|\bevar\b|"
    r"\brupture\b|\baneurysm repair\b|\bsmoking\b",
    re.I,
)


def is_guideline_document(document_type: Any) -> bool:
    """True for official guideline / public-health recommendation sources."""
    return _as_text(document_type).strip().lower() in GUIDELINE_DOC_TYPES


def classify_chunk_content(text: str, section_title: Any = None) -> str:
    """Label a chunk so low-value material can be kept but excluded from search.

    Returns one of: clinical, reference, toc, boilerplate, title_only.
    Nothing is deleted here -- every chunk is still written to chunks.json with
    its label; only the retrieval index is filtered.
    """
    body = (text or "").strip()
    if not body:
        return "boilerplate"

    # Structural checks run FIRST. A contents page lists section names such as
    # "Recommendations for research", so a recommendation-word escape hatch
    # placed ahead of this would let table-of-contents rows through.
    #
    # Density, not an absolute run count: token-aware chunks are small, so a
    # contents page yields many chunks holding only one or two dot-leader runs
    # each. On this corpus every contents chunk scores >= 0.23 and every other
    # chunk scores 0.0, so the 0.10 cut has a wide margin either side.
    compact = re.sub(r"\s+", "", body)
    if _DOT_LEADER_RE.search(body):
        dot_chars = sum(m.count(".") for m in _DOT_LEADER_RE.findall(body))
        if dot_chars / (len(compact) or 1) >= 0.10:
            return "toc"

    # Page rules, dividers and form lines ("_ _ _ _ _", "-------"): almost no
    # letters or digits, so they carry no retrievable clinical information.
    # Real clinical text in this corpus sits above 0.20 alphanumeric density.
    if len(compact) >= 20:
        alnum = sum(1 for ch in compact if ch.isalnum())
        if alnum / len(compact) < 0.20:
            return "boilerplate"

    # Never demote real recommendation text (short or table-shaped). This sits
    # ahead of the section-label check on purpose: section titles are inherited
    # and can be wrong, so a genuine graded recommendation must not be dropped
    # merely because it inherited a "REFERENCES" heading.
    if _RECOMMENDATION_RE.search(body) and _CLINICAL_SIGNAL_RE.search(body):
        return "clinical"

    section = _as_text(section_title)
    if _REFERENCE_SECTION_RE.search(section):
        return "reference"
    if len(_CITATION_RE.findall(body)) >= 3:
        return "reference"

    stripped = _norm(body)
    boiler_hits = _BOILERPLATE_RE.findall(body)
    if boiler_hits:
        residual = _BOILERPLATE_RE.sub(" ", body)
        residual = re.sub(r"page\s*\d+\s*of\s*\d*|https?://\S+|www\.\S+", " ", residual, flags=re.I)
        if len(_norm(residual)) < 200 and not _CLINICAL_SIGNAL_RE.search(residual):
            return "boilerplate"

    if len(stripped) < 200 and not _CLINICAL_SIGNAL_RE.search(stripped):
        return "title_only"

    return "clinical"


def build_chunks(pages_df: pd.DataFrame, recs_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    recs_df = recs_df if recs_df is not None else pd.DataFrame()
    for document_id, doc_pages in pages_df.groupby("document_id", sort=True):
        doc_pages = doc_pages.sort_values("page_number")
        buffer_parts: list[dict[str, Any]] = []
        doc_seq = 0

        def flush(force: bool = False) -> None:
            nonlocal buffer_parts, doc_seq
            if not buffer_parts:
                return
            combined = "\n\n".join(p["text"] for p in buffer_parts if (p.get("text") or "").strip()).strip()
            if not combined or len(combined) < MIN_VALID_CHARS:
                buffer_parts = []
                return
            page_start = int(buffer_parts[0]["page_number"])
            page_end = int(buffer_parts[-1]["page_number"])
            meta = buffer_parts[0]
            recs_nearby: list[dict[str, Any]] = []
            for part in buffer_parts:
                recs_nearby.extend(part["recs"])
            pieces = split_text(combined)
            if not pieces:
                pieces = [combined] if combined else []
            section_title = _as_optional(meta["section_title"])
            section_source = _as_optional(meta["section_source"]) or "unknown"
            for i, piece in enumerate(pieces, start=1):
                rec_meta = attach_recommendation_metadata(piece, recs_nearby)
                excerpt = piece[:400]
                doc_seq += 1
                # Sequence is per document so adding a new PDF never renumbers
                # the chunk IDs of documents that were already indexed.
                chunk_id = f"{document_id}__p{page_start}-{page_end}__c{doc_seq:04d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "document_name": meta["document_name"],
                        "document_type": meta["document_type"],
                        "is_guideline": is_guideline_document(meta["document_type"]),
                        "section_title": section_title,
                        "section_source": section_source,
                        "page_number": page_start,
                        "page_start": page_start,
                        "page_end": page_end,
                        "source_file": meta["source_file"],
                        "chunk_text": piece,
                        "source_excerpt": excerpt,
                        "token_count": count_tokens(piece),
                        "char_count": len(piece),
                        "content_type": classify_chunk_content(piece, section_title),
                        "recommendation_id": rec_meta["recommendation_id"],
                        "recommendation_grade": rec_meta["recommendation_grade"],
                        "evidence_level": rec_meta["evidence_level"],
                    }
                )
            buffer_parts = []

        current_section = object()
        for _, row in doc_pages.iterrows():
            text = page_working_text(row)
            status = row.get("extraction_status")
            if status in {"corrupted"}:
                flush()
                continue
            section = _as_optional(row.get("section_title"))
            recs = recs_for_page(recs_df, row["document_id"], int(row["page_number"]))
            item = {
                "text": text,
                "page_number": int(row["page_number"]),
                "document_name": row.get("document_name"),
                "document_type": row.get("document_type"),
                "section_title": section,
                "section_source": _as_optional(row.get("section_source")),
                "source_file": row.get("source_file"),
                "recs": recs,
            }
            # Sparse title slides stay as their own chunk so they are not mixed into clinical body text.
            if status == "low_text" or len(text) < SHORT_WARN_CHARS:
                flush()
                buffer_parts = [item]
                flush()
                current_section = section
                continue
            if section != current_section and buffer_parts:
                flush()
            current_section = section
            tentative = ("\n\n".join(p["text"] for p in buffer_parts + [item])).strip()
            if buffer_parts and len(tentative) > MAX_CHARS:
                flush()
            buffer_parts.append(item)
            packed = "\n\n".join(p["text"] for p in buffer_parts)
            if len(packed) >= TARGET_CHARS:
                flush()
        flush()
    return chunks


def token_statistics(chunks: list[dict[str, Any]], model_name: str = DEFAULT_EMBED_MODEL) -> dict[str, Any]:
    """Token distribution measured with the embedding model's own tokenizer."""
    limit = model_token_limit(model_name)
    counts = sorted(
        int(c.get("token_count") if c.get("token_count") is not None else count_tokens(c.get("chunk_text") or "", model_name))
        for c in chunks
    )
    if not counts:
        return {
            "model_name": model_name,
            "token_limit": limit,
            "n_chunks": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "exceeding_limit": 0,
        }
    n = len(counts)
    mid = n // 2
    median = float(counts[mid]) if n % 2 else (counts[mid - 1] + counts[mid]) / 2
    return {
        "model_name": model_name,
        "token_limit": limit,
        "n_chunks": n,
        "min_tokens": counts[0],
        "max_tokens": counts[-1],
        "mean_tokens": round(sum(counts) / n, 2),
        "median_tokens": median,
        "exceeding_limit": sum(1 for c in counts if c > limit),
    }


def validate_chunks(chunks: list[dict[str, Any]], model_name: str = DEFAULT_EMBED_MODEL) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    token_limit = model_token_limit(model_name)
    required = [
        "chunk_id",
        "document_id",
        "document_name",
        "document_type",
        "section_title",
        "section_source",
        "page_number",
        "page_start",
        "page_end",
        "source_file",
        "chunk_text",
        "source_excerpt",
    ]
    seen_ids: set[str] = set()
    text_hashes: Counter[str] = Counter()
    invalid_ids: list[str] = []
    valid = 0

    for chunk in chunks:
        issues = []
        for key in required:
            if key not in chunk:
                issues.append(f"missing field {key}")
        cid = chunk.get("chunk_id")
        if not cid:
            issues.append("missing chunk_id")
        elif cid in seen_ids:
            issues.append("duplicate chunk_id")
        else:
            seen_ids.add(cid)
        if not chunk.get("document_id"):
            issues.append("missing document_id")
        if chunk.get("page_number") in (None, ""):
            issues.append("missing page_number")
        if not chunk.get("source_file"):
            issues.append("missing source_file")
        text = chunk.get("chunk_text") or ""
        if not text.strip():
            issues.append("empty chunk")
        elif len(text.strip()) < MIN_VALID_CHARS:
            issues.append("extremely short")
        if len(text) > MAX_CHARS + 200:
            issues.append("excessively large")
        tokens = chunk.get("token_count")
        if tokens is None:
            tokens = count_tokens(text)
        if int(tokens) > token_limit:
            issues.append(f"exceeds embedding token limit ({tokens} > {token_limit})")
        if GLUED_RANGE_RE.search(text):
            issues.append("broken numeric range")
        if CORRUPT_RE.search(text):
            issues.append("corrupted text")
        digest = hashlib.md5(_norm(text).encode("utf-8")).hexdigest()
        text_hashes[digest] += 1
        if chunk.get("section_source") not in {"detected", "inherited", "unknown", None}:
            warnings.append(f"{cid}: unexpected section_source")
        if issues:
            invalid_ids.append(cid or "<missing>")
            errors.append(f"{cid}: " + "; ".join(issues))
        else:
            valid += 1

    duplicate_texts = sum(1 for n in text_hashes.values() if n > 1)
    tokens = token_statistics(chunks, model_name)
    content_types = dict(Counter(str(c.get("content_type") or "unknown") for c in chunks))
    missing_section = sum(1 for c in chunks if _as_optional(c.get("section_title")) is None)
    report = {
        "total": len(chunks),
        "valid": valid,
        "invalid": len(invalid_ids),
        "duplicates": duplicate_texts,
        "missing_metadata": sum(1 for e in errors if "missing" in e),
        "missing_section_title": missing_section,
        "content_types": content_types,
        "tokens": tokens,
        "invalid_chunk_ids": invalid_ids,
        "errors": errors[:50],
        "warnings": warnings[:30],
        "status": "PASS" if not invalid_ids and tokens["exceeding_limit"] == 0 else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report


def save_chunks(chunks: list[dict[str, Any]], report: dict[str, Any], project_root: Path | None = None) -> Path:
    project_root = project_root or find_project_root()
    out_dir = project_root / "data" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "chunks.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality": report,
        "chunks": chunks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


# Content labels excluded from the retrieval index. Chunks are still written to
# chunks.json with their label -- this filters search, it does not delete data.
NON_RETRIEVABLE_CONTENT_TYPES = {"reference", "toc", "boilerplate", "title_only"}


def embeddable_chunks(
    chunks: list[dict[str, Any]],
    quality: dict[str, Any],
    guidelines_only: bool = True,
    drop_content_types: set[str] | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> list[dict[str, Any]]:
    """Select the chunks that go into the vector index.

    Excludes invalid/corrupt chunks, non-informative content (references, table
    of contents, title-only slides, boilerplate) and, by default, documents that
    are not official guidelines. Anything still over the token limit is refused
    rather than silently truncated at encode time.
    """
    invalid = set(quality.get("invalid_chunk_ids") or [])
    drop = NON_RETRIEVABLE_CONTENT_TYPES if drop_content_types is None else drop_content_types
    limit = model_token_limit(model_name)
    out = []
    for chunk in chunks:
        text = (chunk.get("chunk_text") or "").strip()
        if not text:
            continue
        if chunk.get("chunk_id") in invalid:
            continue
        if CORRUPT_RE.search(text):
            continue
        content_type = str(chunk.get("content_type") or classify_chunk_content(text, chunk.get("section_title")))
        if content_type in drop:
            continue
        if guidelines_only and not chunk.get("is_guideline", is_guideline_document(chunk.get("document_type"))):
            continue
        tokens = chunk.get("token_count")
        if tokens is None:
            tokens = count_tokens(text, model_name)
        if int(tokens) > limit:
            raise ValueError(
                f"{chunk.get('chunk_id')} has {tokens} tokens, over the {limit}-token "
                f"limit of {model_name}; re-run chunking before embedding."
            )
        out.append(chunk)
    return out


def describe_chunk(chunk: dict[str, Any], max_chars: int | None = None) -> str:
    """Full inspection view of a chunk: metadata AND the actual text."""
    text = chunk.get("chunk_text") or ""
    body = text if max_chars is None or len(text) <= max_chars else text[:max_chars] + " ..."
    section = _as_optional(chunk.get("section_title")) or "<none>"
    return (
        f"chunk_id     : {chunk.get('chunk_id')}\n"
        f"document     : {chunk.get('document_name')} ({chunk.get('document_id')})\n"
        f"is_guideline : {chunk.get('is_guideline')}\n"
        f"page         : {chunk.get('page_number')} (pages {chunk.get('page_start')}-{chunk.get('page_end')})\n"
        f"section      : {section} [{chunk.get('section_source')}]\n"
        f"content_type : {chunk.get('content_type')}\n"
        f"tokens       : {chunk.get('token_count')}   chars: {chunk.get('char_count') or len(text)}\n"
        f"text         :\n{body}\n"
    )


def print_chunk_samples(
    chunks: list[dict[str, Any]],
    per_document: int = 2,
    max_chars: int | None = 700,
) -> None:
    """Print real chunk text (not just metadata) for manual inspection."""
    seen: Counter[str] = Counter()
    for chunk in chunks:
        doc = str(chunk.get("document_id"))
        if seen[doc] >= per_document:
            continue
        seen[doc] += 1
        print("-" * 88)
        print(describe_chunk(chunk, max_chars=max_chars))


# Which chunker `run_chunking` ships. "atomic" is the production default adopted
# after Experiment 12 and the final corrected validation on final20; it cuts at
# structural anchors and keeps recommendations whole (ingestion.atomic_chunking).
# "page_buffer" is the historical chunker preserved in `build_chunks` above --
# it is what produced data/archive_baseline_index/ and every pre-Experiment-12
# evaluation, and it is kept so those results stay reproducible.
DEFAULT_CHUNKER = "atomic"


def build_chunks_for(strategy: str, pages_df: pd.DataFrame, recs_df: pd.DataFrame | None):
    if strategy == "page_buffer":
        return build_chunks(pages_df, recs_df)
    if strategy == "atomic":
        import ingestion.atomic_chunking as cac  # local import avoids a circular import

        return cac.build_chunks(pages_df, recs_df)
    raise ValueError(f"Unknown chunker strategy: {strategy}")


def run_chunking(project_root: Path | None = None, strategy: str = DEFAULT_CHUNKER) -> dict[str, Any]:
    loaded = load_processed(project_root)
    chunks = build_chunks_for(strategy, loaded["pages_df"], loaded["recommendations_df"])
    quality = validate_chunks(chunks)
    path = save_chunks(chunks, quality, loaded["project_root"])
    return {
        "project_root": loaded["project_root"],
        "chunks": chunks,
        "quality": quality,
        "path": path,
        "relative_path": relative_posix(path, loaded["project_root"]),
        "strategy": strategy,
    }


def print_chunk_report(quality: dict[str, Any]) -> None:
    print("CHUNKS")
    print(f"- total: {quality['total']}")
    print(f"- valid: {quality['valid']}")
    print(f"- invalid: {quality['invalid']}")
    print(f"- duplicates: {quality['duplicates']}")
    print(f"- missing metadata issues: {quality['missing_metadata']}")
    print(f"- missing section_title: {quality.get('missing_section_title')}")
    tokens = quality.get("tokens") or {}
    if tokens:
        print(f"\nTOKENS (tokenizer: {tokens.get('model_name')}, limit {tokens.get('token_limit')})")
        print(f"- chunks: {tokens.get('n_chunks')}")
        print(f"- min: {tokens.get('min_tokens')}")
        print(f"- median: {tokens.get('median_tokens')}")
        print(f"- mean: {tokens.get('mean_tokens')}")
        print(f"- max: {tokens.get('max_tokens')}")
        print(f"- EXCEEDING_LIMIT = {tokens.get('exceeding_limit')}")
    if quality.get("content_types"):
        print("\nCONTENT TYPES")
        for name, count in sorted(quality["content_types"].items()):
            print(f"- {name}: {count}")
    print(f"\n- status: {quality['status']}")
    if quality["errors"]:
        print("\nERRORS (first 20):")
        for err in quality["errors"][:20]:
            print(" -", err)
    if quality["warnings"]:
        print("\nWARNINGS:")
        for warn in quality["warnings"][:20]:
            print(" -", warn)
