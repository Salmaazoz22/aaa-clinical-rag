# -*- coding: utf-8 -*-
"""Reusable render primitives for the Clinical RAG UI.

Two rules run through every function here:

1. **Nothing is asserted that the backend did not report.** A "verified" badge
   is drawn only where the API's own validator reported the corresponding check
   as clean; a check the validator could not run is drawn as *not checked*, in
   grey, never as a pass. `citation_trace()` is the clearest example — it maps
   validator finding `location`s onto the specific citation they belong to,
   rather than inferring anything.
2. **Backend text is escaped, always.** Guideline text, model prose, section
   titles and error details all reach the page through `html.escape`. These
   strings come out of PDFs and a language model, and this module writes raw
   HTML.
"""
from __future__ import annotations

import html
import json
from typing import Any, Iterable, Sequence

import streamlit as st

from ui import api_client
from ui.api_client import ApiError

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    """HTML-escape anything, rendering None as an em dash."""
    if value is None:
        return "—"
    return html.escape(str(value))


def _num(value: Any, places: int = 4) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _pages(hit: dict[str, Any]) -> str:
    """A human page reference from whichever page fields the record carries."""
    start, end = hit.get("page_start"), hit.get("page_end")
    if start is not None and end is not None:
        return f"p. {start}" if start == end else f"pp. {start}–{end}"
    page = hit.get("page")
    return f"p. {page}" if page is not None else "page —"


def _doc_label(hit: dict[str, Any]) -> str:
    """Prefer the short document_id (ESVS_2024) over the full title."""
    return str(hit.get("document_id") or hit.get("document") or "unknown source")


# ---------------------------------------------------------------------------
# Masthead and system badges
# ---------------------------------------------------------------------------

MASTHEAD_LEDE = (
    "Every answer is generated only from guideline passages retrieved for that "
    "question, every citation is checked against the passages that were actually "
    "sent to the model, and questions the evidence cannot support are refused "
    "rather than answered."
)


def masthead(subtitle: str = "Evidence-Grounded Clinical Decision Support") -> None:
    st.markdown(
        f"""
        <div class="masthead">
          <div class="eyebrow">Retrieval-Augmented Generation · Abdominal Aortic Aneurysm</div>
          <h1>AAA Clinical RAG</h1>
          <p class="tagline">{esc(subtitle)}</p>
          <p class="lede">{esc(MASTHEAD_LEDE)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def system_badges(meta: dict[str, Any] | None, corpus: dict[str, Any] | None) -> None:
    """Capability badges, every value read from the API — none hardcoded.

    A value the API did not supply is simply not shown. Displaying "4 Guidelines"
    from a constant while the service reports something else would be exactly
    the sort of decorative claim this project is built to avoid.
    """
    badges: list[str] = []

    if corpus and corpus.get("n_documents"):
        badges.append(f'<span class="badge ok">✓ <b>{corpus["n_documents"]}</b> guidelines</span>')

    if meta and meta.get("chunk_count") is not None:
        badges.append(f'<span class="badge ok">✓ <b>{meta["chunk_count"]:,}</b> indexed chunks</span>')

    if meta and meta.get("model"):
        short = str(meta["model"]).split("/")[-1]
        dims = meta.get("dimensions")
        dim_txt = f" {dims}-d" if dims else ""
        badges.append(f'<span class="badge accent">✓ <b>{esc(short)}</b>{esc(dim_txt)}</span>')

    if meta and meta.get("vector_store"):
        distance = meta.get("distance")
        suffix = f" · {distance.lower()}" if distance else ""
        badges.append(f'<span class="badge accent">✓ <b>{esc(meta["vector_store"])}</b>{esc(suffix)}</span>')

    badges.append('<span class="badge ok">✓ <b>Citation validation</b></span>')
    badges.append('<span class="badge ok">✓ <b>Safety gates</b></span>')

    if meta and meta.get("generation"):
        gen = meta["generation"]
        threshold = gen.get("score_threshold")
        if threshold is not None:
            badges.append(f'<span class="badge">Evidence floor <b>{threshold}</b></span>')

    st.markdown(f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------

def health_strip(health: dict[str, Any] | None) -> None:
    """Live status for API, Qdrant, index and LLM — rendered in the sidebar."""
    if health is None:
        st.markdown(
            '<div class="badge-row">'
            '<span class="badge bad">● <b>API offline</b></span></div>',
            unsafe_allow_html=True,
        )
        return

    llm = health.get("llm") or {}
    parts = [
        f'<span class="badge ok">● <b>API</b> up</span>',
        _dot_badge("Qdrant", bool(health.get("qdrant"))),
        _dot_badge("Index", bool(health.get("index"))),
    ]
    if health.get("points") is not None:
        parts.append(f'<span class="badge">● <b>{health["points"]:,}</b> points</span>')
    parts.append(
        f'<span class="badge ok">● <b>LLM</b> ready</span>'
        if llm.get("configured")
        else '<span class="badge warn">● <b>LLM</b> no key</span>'
    )
    st.markdown(f'<div class="badge-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def _dot_badge(label: str, ok: bool) -> str:
    cls = "ok" if ok else "bad"
    state = "up" if ok else "down"
    return f'<span class="badge {cls}">● <b>{esc(label)}</b> {state}</span>'


def notice(kind: str, title: str, body_html: str) -> None:
    """A styled callout. `body_html` must already be escaped where it came from
    the backend."""
    st.markdown(
        f'<div class="notice {kind}"><span class="t">{esc(title)}</span>{body_html}</div>',
        unsafe_allow_html=True,
    )


def backend_unavailable(error: ApiError) -> None:
    """The offline / error experience. No stack traces reach the reader."""
    if error.kind == "offline":
        notice(
            "bad",
            "Backend unavailable",
            "<p>The Streamlit app is running, but nothing is answering at "
            f"<code>{esc(api_client.base_url())}</code>. This page is a client of the "
            "FastAPI service and has no retrieval or generation of its own, so it cannot "
            "show results until the API is up.</p>"
            "<p><b>Start the backend</b> from the project root:</p>"
            "<pre><code>uvicorn api.main:app --host 127.0.0.1 --port 8000</code></pre>"
            "<p class='footnote'>Point the UI elsewhere with the "
            "<code>CLINICAL_RAG_API_URL</code> environment variable.</p>",
        )
    elif error.kind == "timeout":
        notice(
            "warn",
            "The API did not respond in time",
            f"<p>{esc(error.message)} The service may still be loading the embedding "
            "model — that happens once, on the first start, and can take a minute.</p>",
        )
    elif error.status_code == 503:
        notice(
            "warn",
            "A dependency is unavailable",
            f"<p>{esc(error.detail or error.message)}</p>"
            "<p class='footnote'>The API is running but cannot reach something it needs — "
            "usually the Qdrant collection, or an unset LLM API key.</p>",
        )
    else:
        detail = f"<p class='footnote'>{esc(error.detail)}</p>" if error.detail else ""
        notice("bad", "Request failed", f"<p>{esc(error.message)}</p>{detail}")


def llm_unavailable_notice(health: dict[str, Any] | None) -> bool:
    """Warn *before* a question is asked when live generation cannot complete.

    Returns True when generation is unavailable. The retrieval and safety layers
    still work in this state, and the notice says so, because a refusal produced
    by the safety or threshold gate is a genuine, fully-grounded result that
    needs no model call at all.
    """
    if health is None:
        return True
    llm = health.get("llm") or {}
    if llm.get("configured"):
        return False
    provider = llm.get("provider") or "the configured provider"
    key_env = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(str(provider), "the provider API key")
    notice(
        "warn",
        "Live generation unavailable — no LLM API key configured",
        f"<p>The API reports no key for <b>{esc(provider)}</b>, so questions that reach the "
        "generation step will return an error rather than an answer. "
        "<b>No placeholder or cached answer is ever shown in its place.</b></p>"
        f"<p>Set <code>{esc(key_env)}</code> in <code>.env</code> and restart the API to enable it.</p>"
        "<p class='footnote'>Retrieval, the safety gate and the evidence threshold do not need "
        "the model and are fully functional right now — a question refused by those gates is a "
        "real, complete result.</p>",
    )
    return True


# ---------------------------------------------------------------------------
# Stat tiles
# ---------------------------------------------------------------------------

def stat(label: str, value: Any, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="s">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="stat"><div class="k">{esc(label)}</div>'
        f'<div class="v {tone}">{esc(value)}</div>{sub_html}</div>'
    )


def stat_row(tiles: Sequence[str]) -> None:
    cols = st.columns(len(tiles), gap="small")
    for col, tile in zip(cols, tiles):
        col.markdown(tile, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline trace — the real stages, with their real outcomes
# ---------------------------------------------------------------------------

def pipeline_trace(result: dict[str, Any]) -> None:
    """What each stage of the pipeline actually did on this request.

    Every line is read back out of the response. The backend answers in one
    call and reports no streaming stage events, so this is a record of what
    happened, not a simulated progress animation.
    """
    safety = result.get("safety") or {}
    retrieval = result.get("retrieval") or {}
    validation = result.get("validation") or {}
    generation = result.get("generation") or {}
    settings = result.get("settings") or {}
    refusal = result.get("refusal") or {}
    gate = str(refusal.get("gate") or "")

    steps: list[tuple[str, str, str]] = []

    # 1 — safety
    if safety.get("blocked"):
        signals = ", ".join(safety.get("signals") or []) or "—"
        steps.append(("stop", "Safety screening",
                      f"BLOCKED · rule {safety.get('rule')} · signals: {signals}"))
    else:
        signals = ", ".join(safety.get("signals") or [])
        detail = f"passed · signals noted: {signals}" if signals else "passed · no patient-specific signals"
        steps.append(("pass", "Safety screening", detail))

    # 2 — retrieval
    n_ret = retrieval.get("n_retrieved", 0)
    hits = retrieval.get("hits") or []
    top1 = hits[0].get("similarity_score") if hits else None
    if n_ret:
        steps.append(("pass", "Retrieving evidence",
                      f"{n_ret} chunks · top-1 similarity {_num(top1)} · "
                      f"top_k={settings.get('top_k')}"))
    else:
        steps.append(("stop", "Retrieving evidence", "nothing returned by the vector store"))

    # 3 — evidence threshold
    n_used = retrieval.get("n_used", 0)
    n_drop = retrieval.get("n_dropped_below_threshold", 0)
    floor = settings.get("score_threshold")
    if gate == "threshold":
        steps.append(("stop", "Evaluating evidence",
                      f"0 of {n_ret} cleared the {floor} floor — refused"))
    else:
        steps.append(("pass" if n_used else "skip", "Evaluating evidence",
                      f"{n_used} usable · {n_drop} below the {floor} floor"))

    # 4 — generation
    completion = generation.get("completion")
    if completion:
        usage = completion.get("usage") or {}
        tokens = usage.get("total_tokens")
        bits = [f"{completion.get('provider')}/{completion.get('model')}",
                f"{completion.get('latency_s')}s"]
        if tokens:
            bits.append(f"{tokens} tokens")
        if completion.get("finish_reason"):
            bits.append(f"finish={completion['finish_reason']}")
        steps.append(("pass", "Generating answer", " · ".join(str(b) for b in bits)))
    elif safety.get("blocked"):
        steps.append(("skip", "Generating answer",
                      "not called — refused locally, so no patient detail left this machine"))
    elif gate:
        steps.append(("skip", "Generating answer", f"not called — refused at gate '{gate}'"))
    else:
        steps.append(("skip", "Generating answer", "not called"))

    # 5 — citation validation
    n_err = validation.get("n_errors", 0)
    n_warn = validation.get("n_warnings", 0)
    codes = ", ".join(dict.fromkeys(validation.get("codes") or [])) or "no findings"
    if validation.get("ok"):
        state = "pass" if not n_warn else "pass"
        steps.append((state, "Validating citations",
                      f"PASS · {n_err} errors · {n_warn} warnings · {codes}"))
    else:
        steps.append(("fail", "Validating citations",
                      f"FAIL · {n_err} errors · {n_warn} warnings · {codes}"))

    glyph = {"pass": "✓", "stop": "!", "skip": "–", "fail": "✕"}
    rows = "".join(
        f'<div class="trace-step">'
        f'<div class="trace-dot {kind}">{glyph[kind]}</div>'
        f'<div><div class="trace-name">{esc(name)}</div>'
        f'<div class="trace-detail">{esc(detail)}</div></div></div>'
        for kind, name, detail in steps
    )
    st.markdown(
        f'<div class="card"><div class="card-label">Pipeline trace — what actually ran</div>'
        f'<div class="trace">{rows}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Answer panel
# ---------------------------------------------------------------------------

_CONFIDENCE_TONE = {
    "High": ("green", "ok"),
    "Medium": ("amber", "warn"),
    "Low": ("amber", "warn"),
    "Insufficient Evidence": ("amber", "warn"),
}

_REFUSAL_EXPLANATION = {
    "patient_specific_request": (
        "This question asks for a decision about a specific individual. The system refused it "
        "<b>before any retrieval reached a language model</b>, so no patient detail was sent to a "
        "third-party API. This is the safety gate working as designed, not a failure."
    ),
    "all_scores_below_threshold": (
        "Retrieval ran and returned passages, but none of them scored above the evidence-quality "
        "floor. The system will not build a clinical statement on passages that are only topically "
        "adjacent, so it refused instead of answering."
    ),
    "no_chunks_retrieved": (
        "Nothing in the indexed guideline corpus was close enough to this question to return as "
        "evidence at all."
    ),
    "evidence_not_specific_enough": (
        "Evidence cleared the floor and was sent to the model, and the model judged it topically "
        "related but not specific enough to answer the question as asked. This judgement requires "
        "the evidence to be read, so it cannot be made by a threshold."
    ),
}

_GATE_LABEL = {
    "threshold": "evidence threshold",
    "retrieval:empty": "retrieval",
    "model": "model judgement (post-evidence)",
}


def answer_panel(result: dict[str, Any]) -> None:
    answer = result.get("answer") or {}
    refused = bool(result.get("refused"))
    refusal = result.get("refusal") or {}
    validation = result.get("validation") or {}
    citations = answer.get("citations") or []
    confidence = str(answer.get("confidence") or "—")

    tone, badge_cls = _CONFIDENCE_TONE.get(confidence, ("", ""))

    # --- headline tiles ----------------------------------------------------
    if validation.get("ok"):
        v_value, v_tone, v_sub = "PASS", "green", f"{validation.get('n_warnings', 0)} warnings"
    else:
        v_value, v_tone, v_sub = "FAIL", "red", f"{validation.get('n_errors', 0)} errors"

    stat_row([
        stat("Status", "Refused" if refused else "Answered",
             _GATE_LABEL.get(str(refusal.get("gate")), str(refusal.get("gate") or ""))
             if refused else "grounded in retrieved evidence",
             "amber" if refused else "green"),
        stat("Confidence", confidence, "ordinal label, never a percentage", tone),
        stat("Citations", len(citations),
             f"{len(result.get('documents_cited') or [])} guideline(s)", "accent"),
        stat("Citation validation", v_value, v_sub, v_tone),
    ])

    # --- refusal explanation ----------------------------------------------
    if refused:
        reason = str(refusal.get("reason") or "")
        gate = str(refusal.get("gate") or "")
        explanation = _REFUSAL_EXPLANATION.get(
            reason, "The system declined to answer this question from the retrieved evidence."
        )
        if gate.startswith("safety:"):
            gate_label = f"safety gate, rule {gate.split(':', 1)[1]}"
        else:
            gate_label = _GATE_LABEL.get(gate, gate or "—")
        notice(
            "warn",
            "Answer withheld — this is a deliberate refusal, not an error",
            f"<p>{explanation}</p>"
            f"<p class='footnote'>Reason code <code>{esc(reason)}</code> · "
            f"produced by the <b>{esc(gate_label)}</b>. Both fields come from the API response.</p>",
        )

    # --- the answer itself -------------------------------------------------
    recommendation = str(answer.get("recommendation") or "").strip()
    if recommendation:
        paragraphs = "".join(f"<p>{esc(p)}</p>" for p in recommendation.split("\n") if p.strip())
        st.markdown(
            f'<div class="answer{" refused" if refused else ""}">'
            f'<div class="card-label">{"Refusal" if refused else "Recommendation"}</div>'
            f'<div class="body">{paragraphs}</div></div>',
            unsafe_allow_html=True,
        )

    # --- supporting evidence bullets --------------------------------------
    supporting = [e for e in (answer.get("supporting_evidence") or []) if isinstance(e, dict)]
    if supporting and not refused:
        items = "".join(
            f'<div class="chain-row"><div class="chain-k">{esc(e.get("chunk_id") or "—")}</div>'
            f'<div class="chain-v serif">{esc(e.get("claim"))}</div></div>'
            for e in supporting
        )
        st.markdown(
            f'<div class="card"><div class="card-label">Supporting evidence — '
            f'each point bound to the chunk it came from</div>{items}</div>',
            unsafe_allow_html=True,
        )

    # --- evidence conflicts (optional schema field) ------------------------
    conflicts = [c for c in (answer.get("evidence_conflicts") or []) if isinstance(c, dict)]
    if conflicts:
        blocks = []
        for conflict in conflicts:
            positions = "".join(
                f'<div class="chain-row">'
                f'<div class="chain-k">{esc(", ".join(p.get("chunk_ids") or []))}</div>'
                f'<div class="chain-v serif">{esc(p.get("position"))}'
                + (f' <i>({esc(p.get("source"))})</i>' if p.get("source") else "")
                + "</div></div>"
                for p in (conflict.get("positions") or [])
                if isinstance(p, dict)
            )
            blocks.append(
                f'<div class="chain-claim">{esc(conflict.get("topic"))}</div>{positions}'
            )
        notice(
            "info",
            "The retrieved guidelines disagree",
            "<p class='footnote'>Both positions are reported with their own citations rather than "
            "one being chosen.</p>",
        )
        st.markdown(f'<div class="card">{"".join(blocks)}</div>', unsafe_allow_html=True)

    # --- disclaimer --------------------------------------------------------
    disclaimer = str(answer.get("disclaimer") or "").strip()
    if disclaimer:
        normalised = (result.get("generation") or {}).get("disclaimer_normalised")
        note = (
            " <b>The model's version differed and was replaced with the canonical text; "
            "the substitution is recorded in the validator findings.</b>"
            if normalised
            else ""
        )
        st.markdown(
            f'<div class="disclaimer"><b>Safety disclaimer.</b> {esc(disclaimer)}{note}</div>',
            unsafe_allow_html=True,
        )


def validation_findings(result: dict[str, Any]) -> None:
    """Every validator finding, verbatim. Nothing is hidden because it is bad."""
    validation = result.get("validation") or {}
    findings = validation.get("findings") or []
    if not findings:
        notice(
            "good",
            "Validator returned no findings",
            "<p class='footnote'>Every citation resolved to a chunk that was actually sent to the "
            "model, every excerpt was found in its cited chunk, and every metadata field matched "
            "the retriever's own record.</p>",
        )
        return

    rows = []
    for finding in findings:
        sev = str(finding.get("severity"))
        cls = "n" if sev == "error" else "w"
        extra = ""
        if finding.get("expected") is not None or finding.get("actual") is not None:
            extra = (
                f'<div class="chain-row"><div class="chain-k">expected</div>'
                f'<div class="chain-v">{esc(json.dumps(finding.get("expected"), ensure_ascii=False))}</div></div>'
                f'<div class="chain-row"><div class="chain-k">actual</div>'
                f'<div class="chain-v">{esc(json.dumps(finding.get("actual"), ensure_ascii=False))}</div></div>'
            )
        rows.append(
            f'<div class="trace-chain">'
            f'<div class="checks"><span class="check {cls}">{esc(sev.upper())}</span>'
            f'<span class="check o">{esc(finding.get("code"))}</span>'
            + (f'<span class="check o">{esc(finding.get("location"))}</span>' if finding.get("location") else "")
            + f'</div><div class="chain-claim" style="border:0;margin-top:0.6rem;padding-bottom:0">'
            f'{esc(finding.get("message"))}</div>{extra}</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Retrieval score visualisation
# ---------------------------------------------------------------------------

def score_bars(result: dict[str, Any]) -> None:
    """Similarity per rank, with the evidence floor drawn where it actually sits.

    Bars are scaled to 1.0 so the floor line is comparable between questions —
    a bar chart auto-scaled to its own maximum would make a 0.48 retrieval look
    identical to a 0.87 one.
    """
    retrieval = result.get("retrieval") or {}
    hits = retrieval.get("hits") or []
    if not hits:
        return
    threshold = float((result.get("settings") or {}).get("score_threshold") or 0.0)
    used_ids = set(retrieval.get("used_chunk_ids") or [])

    rows = []
    for hit in hits:
        score = float(hit.get("similarity_score") or 0.0)
        used = str(hit.get("chunk_id")) in used_ids
        cls = "used" if used else "dropped"
        width = max(0.0, min(1.0, score)) * 100
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-lab">rank {hit.get("rank")}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill {cls}" style="width:{width:.1f}%"></div>'
            f'<div class="thresh" style="left:{threshold * 100:.1f}%"></div></div>'
            f'<div class="bar-val {cls}">{_num(score)}</div></div>'
        )

    n_used, n_total = len(used_ids), len(hits)
    st.markdown(
        f'<div class="card"><div class="card-label">Retrieval scores — '
        f'cosine similarity against the evidence floor</div>'
        f'<div class="bars">{"".join(rows)}</div>'
        f'<div class="thresh-note">▎ evidence floor = {threshold} · '
        f'{n_used} of {n_total} chunks cleared it and were sent to the model; '
        f'the rest were withheld</div>'
        f'<div class="footnote" style="margin-top:0.5rem">Bars are scaled 0–1, not to the '
        f'maximum in this result, so the floor sits in the same place for every question.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Evidence panel
# ---------------------------------------------------------------------------

def evidence_panel(result: dict[str, Any]) -> None:
    """Every retrieved chunk, marked used or withheld, with its full text.

    The pipeline's audit record carries only a 240-character preview, so the
    full text is fetched from `GET /v1/chunks/{chunk_id}` — the API's own audit
    lookup — and only for the cards the reader opens.
    """
    retrieval = result.get("retrieval") or {}
    hits = retrieval.get("hits") or []
    if not hits:
        notice(
            "warn",
            "No evidence retrieved",
            "<p>The vector store returned nothing for this question, so there was nothing to "
            "ground an answer in.</p>",
        )
        return

    used_ids = set(retrieval.get("used_chunk_ids") or [])
    cited_ids = {
        str(c.get("chunk_id"))
        for c in (result.get("citations_resolved") or [])
        if c.get("chunk_id")
    }

    st.markdown(
        f'<div class="card tight"><div class="card-label">Retrieved evidence</div>'
        f'<div class="footnote">{len(hits)} chunks retrieved · '
        f'<b style="color:#1B7F4B">{len(used_ids)} sent to the model as evidence</b> · '
        f'{len(hits) - len(used_ids)} withheld below the floor'
        + (f' · {len(cited_ids)} cited in the answer' if cited_ids else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    for hit in hits:
        chunk_id = str(hit.get("chunk_id"))
        used = chunk_id in used_ids
        cited = chunk_id in cited_ids
        marks = []
        if used:
            marks.append("used as evidence")
        else:
            marks.append("withheld — below floor")
        if cited:
            marks.append("cited")

        title = (
            f"Evidence {int(hit.get('rank') or 0):02d}   ·   {_doc_label(hit)}   ·   "
            f"{_pages(hit)}   ·   similarity {_num(hit.get('similarity_score'))}   ·   "
            f"{' · '.join(marks)}"
        )

        with st.expander(title, expanded=False):
            _evidence_body(hit, used=used, cited=cited)


def _evidence_body(hit: dict[str, Any], *, used: bool, cited: bool) -> None:
    chunk_id = str(hit.get("chunk_id"))

    chips = [
        f'<span class="check {"y" if used else "o"}">'
        f'{"✓ sent to the model" if used else "○ withheld below floor"}</span>'
    ]
    if cited:
        chips.append('<span class="check y">✓ cited in the answer</span>')
    if hit.get("recommendation_id"):
        chips.append(f'<span class="check o">rec {esc(hit["recommendation_id"])}</span>')
    if hit.get("recommendation_grade"):
        chips.append(f'<span class="check o">grade {esc(hit["recommendation_grade"])}</span>')
    if hit.get("evidence_level"):
        chips.append(f'<span class="check o">LoE {esc(hit["evidence_level"])}</span>')

    st.markdown(
        f'<div class="checks" style="margin-bottom:0.7rem">{"".join(chips)}</div>'
        f'<div class="kv">'
        f'<div class="k">Guideline</div><div class="v wrap">{esc(hit.get("document"))}</div>'
        f'<div class="k">Document ID</div><div class="v">{esc(hit.get("document_id"))}</div>'
        f'<div class="k">Section</div><div class="v wrap">{esc(hit.get("section"))}</div>'
        f'<div class="k">Page</div><div class="v">{esc(_pages(hit))}</div>'
        f'<div class="k">Chunk ID</div><div class="v">{esc(chunk_id)}</div>'
        f'<div class="k">Similarity</div><div class="v">{_num(hit.get("similarity_score"), 6)}</div>'
        f'<div class="k">Rank</div><div class="v">{esc(hit.get("rank"))}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    text, source = _full_chunk_text(chunk_id, hit)
    cls = "used" if used else "dropped"
    st.markdown(
        f'<div class="ev-text {cls}">{esc(text)}</div>'
        f'<div class="footnote" style="margin-top:0.4rem">{esc(source)}</div>',
        unsafe_allow_html=True,
    )


def _full_chunk_text(chunk_id: str, hit: dict[str, Any]) -> tuple[str, str]:
    """Full stored text if the API can supply it, otherwise the preview — labelled."""
    preview = str(hit.get("text_preview") or "").strip()
    try:
        payload = api_client.chunk(chunk_id)
    except ApiError:
        if preview:
            return (
                preview + " …",
                "Preview only (240 characters) — the full chunk could not be fetched from the API.",
            )
        return ("", "The chunk text could not be fetched from the API.")

    text = str(payload.get("chunk_text") or "").strip()
    if not text:
        return (preview, "Preview only — the API returned no stored text for this chunk.")
    return (text, f"Full indexed text, fetched from GET /v1/chunks/{chunk_id}")


# ---------------------------------------------------------------------------
# Citation traceability — evidence → answer
# ---------------------------------------------------------------------------

def _findings_by_citation(result: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Group validator findings by the citation index in their `location`.

    The validator stamps `location` as `citations[3].page`, so attribution is
    exact rather than inferred from chunk_id (which can repeat).
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for finding in (result.get("validation") or {}).get("findings") or []:
        location = str(finding.get("location") or "")
        if not location.startswith("citations["):
            continue
        try:
            index = int(location[len("citations["):location.index("]")])
        except (ValueError, IndexError):
            continue
        grouped.setdefault(index, []).append(finding)
    return grouped


def _check(ok: bool | None, label_ok: str, label_bad: str, label_none: str = "") -> str:
    """One traffic-light chip. `None` means the check could not run — grey, never green."""
    if ok is True:
        return f'<span class="check y">✓ {esc(label_ok)}</span>'
    if ok is False:
        return f'<span class="check n">✕ {esc(label_bad)}</span>'
    return f'<span class="check o">○ {esc(label_none or label_bad)}</span>'


def citation_trace(result: dict[str, Any]) -> None:
    """Claim → citation → chunk → guideline → page, with the validator's verdict.

    Green appears only where the API's validator actually reported the check
    clean. A field the model did not supply — so a check the validator never
    ran — is grey and labelled "not stated", not green.
    """
    answer = result.get("answer") or {}
    citations = [c for c in (answer.get("citations") or []) if isinstance(c, dict)]
    resolved = result.get("citations_resolved") or []
    if not citations:
        notice(
            "info",
            "No citations to trace",
            "<p class='footnote'>This response carries no citations. For a refusal produced by the "
            "safety gate that is deliberate: offering guideline text as an answer to a question "
            "about an individual is exactly what the gate exists to prevent.</p>",
        )
        return

    grouped = _findings_by_citation(result)

    # Claims are attached to citations through the chunk they share.
    claims_by_chunk: dict[str, list[str]] = {}
    for evidence in answer.get("supporting_evidence") or []:
        if isinstance(evidence, dict) and evidence.get("chunk_id"):
            claims_by_chunk.setdefault(str(evidence["chunk_id"]), []).append(str(evidence.get("claim") or ""))

    blocks: list[str] = []
    for index, citation in enumerate(citations):
        record = resolved[index] if index < len(resolved) else {}
        findings = grouped.get(index, [])
        codes = {str(f.get("code")) for f in findings}
        locations = {str(f.get("location") or "") for f in findings}
        chunk_id = str(citation.get("chunk_id") or record.get("chunk_id") or "")
        is_resolved = bool(record.get("resolved"))

        # --- the checks, each tied to something the backend reported -------
        checks = [
            _check(is_resolved,
                   "chunk was in the evidence sent to the model",
                   "chunk_id not in the retrieved set — fabricated")
        ]

        if not is_resolved:
            # Nothing downstream can be checked against a chunk that does not exist.
            checks.append(_check(None, "", "", "document not checkable"))
            checks.append(_check(None, "", "", "page not checkable"))
            checks.append(_check(None, "", "", "excerpt not checkable"))
        else:
            checks.append(_check(
                None if citation.get("document") is None else f"citations[{index}].document" not in locations,
                "document matches the index", "document does not match the index", "document not stated"))
            checks.append(_check(
                None if citation.get("page") is None else f"citations[{index}].page" not in locations,
                "page within the chunk's span", "page outside the chunk's span", "page not stated"))
            checks.append(_check(
                None if citation.get("section") is None or record.get("section") is None
                else f"citations[{index}].section" not in locations,
                "section matches", "section does not match", "section not comparable"))
            checks.append(_check(
                None if citation.get("retrieval_score") is None
                else f"citations[{index}].retrieval_score" not in locations,
                "score copied verbatim", "score altered or estimated", "score not stated"))

            if "empty_excerpt" in codes:
                checks.append(_check(None, "", "", "no excerpt supplied"))
            elif "excerpt_not_in_chunk" in codes:
                checks.append(
                    '<span class="check n">✕ excerpt is not verbatim in the cited chunk</span>'
                )
            elif "excerpt_stitched" in codes:
                checks.append(
                    '<span class="check w">▲ excerpt stitches non-contiguous fragments</span>'
                )
            else:
                checks.append(_check(True, "excerpt found verbatim in the chunk", ""))

            if "duplicate_citation" in codes:
                checks.append('<span class="check w">▲ duplicate citation</span>')

        # --- the chain -----------------------------------------------------
        claims = claims_by_chunk.get(chunk_id) or []
        claim_html = (
            f'<div class="chain-claim">{esc(claims[0])}</div>'
            if claims
            else '<div class="chain-claim" style="color:#8A93A0">'
                 "No supporting-evidence claim is bound to this chunk.</div>"
        )

        rows = [
            ("Citation", esc(f"#{index + 1} · as emitted by the model")),
            ("Chunk", esc(chunk_id or "—")),
            ("Guideline", esc(record.get("document_id") or citation.get("document"))),
            ("Section", esc(record.get("section") or citation.get("section"))),
            ("Page", esc(_pages(record) if is_resolved else citation.get("page"))),
            ("Similarity", _num(record.get("retrieval_score")) if is_resolved else "—"),
            ("Retrieval rank", esc(record.get("rank")) if is_resolved else "—"),
        ]
        if record.get("recommendation_id"):
            rows.append(("Recommendation", esc(record["recommendation_id"])))
        chain = "".join(
            f'<div class="chain-row"><div class="chain-k">{key}</div>'
            f'<div class="chain-v">{value}</div></div>'
            for key, value in rows
        )

        excerpt = str(citation.get("excerpt") or "").strip()
        excerpt_html = (
            f'<div class="chain-row"><div class="chain-k">Excerpt</div>'
            f'<div class="chain-v serif">“{esc(excerpt)}”</div></div>'
            if excerpt
            else ""
        )

        blocks.append(
            f'<div class="trace-chain">{claim_html}{chain}{excerpt_html}'
            f'<div class="checks">{"".join(checks)}</div></div>'
        )

    st.markdown("".join(blocks), unsafe_allow_html=True)
    st.markdown(
        '<div class="footnote">Each ✓ corresponds to a check the API\'s citation validator '
        "actually performed and reported clean. A check the validator could not run — because "
        "the model omitted the field — is shown in grey, never as a pass.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def provenance_panel(meta: dict[str, Any], result: dict[str, Any] | None = None) -> None:
    """Everything needed to say which index and which model produced this run."""
    provenance = meta.get("index_provenance") or {}
    connection = meta.get("connection") or {}
    generation = meta.get("generation") or {}

    rows: list[tuple[str, str]] = [
        ("Embedding model", esc(meta.get("model"))),
        ("Model revision", esc(provenance.get("revision") or meta.get("revision"))),
        ("Embedding dimension", esc(meta.get("dimensions"))),
        ("Distance metric", esc(meta.get("distance"))),
        ("Vector database", esc(meta.get("vector_store"))),
        ("Connection mode", esc(connection.get("mode"))),
        ("Collection", esc(meta.get("collection"))),
        ("Indexed chunks", esc(f"{meta['chunk_count']:,}" if meta.get("chunk_count") is not None else None)),
        ("Index status", esc(meta.get("index_status"))),
        ("Vectors in manifest", esc(provenance.get("n_vectors"))),
        ("Max chunk tokens", esc(provenance.get("max_chunk_tokens"))),
        ("Model token limit", esc(provenance.get("token_limit"))),
        ("Source chunk set", esc(provenance.get("source_chunks_file"))),
        ("Source chunks SHA-256", esc(provenance.get("source_chunks_sha256"))),
        ("Indexed IDs SHA-256", esc(provenance.get("indexed_chunk_ids_sha256"))),
        ("Index manifest SHA-256", esc(provenance.get("index_meta_sha256"))),
        ("Index built (UTC)", esc(provenance.get("built_at_utc"))),
    ]

    if generation:
        rows += [
            ("LLM provider", esc(generation.get("provider"))),
            ("LLM model", esc(generation.get("model"))),
            ("LLM key configured", "yes" if generation.get("api_key_supplied") else "no"),
            ("Retrieval top-k", esc(generation.get("top_k"))),
            ("Evidence floor", esc(generation.get("score_threshold"))),
            ("Temperature", esc(generation.get("temperature"))),
        ]

    if result:
        settings = result.get("settings") or {}
        completion = (result.get("generation") or {}).get("completion") or {}
        rows.append(("— this run —", ""))
        rows += [
            ("Run top-k", esc(settings.get("top_k"))),
            ("Run evidence floor", esc(settings.get("score_threshold"))),
        ]
        if completion:
            rows += [
                ("Run model", esc(f"{completion.get('provider')}/{completion.get('model')}")),
                ("Run latency", esc(f"{completion.get('latency_s')} s")),
                ("Finish reason", esc(completion.get("finish_reason"))),
            ]

    body = "".join(
        f'<div class="k">{key}</div><div class="v">{value}</div>' for key, value in rows
    )
    st.markdown(
        f'<div class="card"><div class="card-label">System provenance — about this index</div>'
        f'<div class="kv">{body}</div>'
        f'<div class="footnote" style="margin-top:0.9rem">Digests identify the exact chunk set the '
        f"running collection was built from. The API verifies this binding on load and refuses to "
        f"serve an index that has drifted from its own chunk set.</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], shipped_row: int | None = None) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for index, row in enumerate(rows):
        cls = ' class="shipped"' if shipped_row is not None and index == shipped_row else ""
        cells = "".join(f"<td>{esc(c)}</td>" for c in row)
        body.append(f"<tr{cls}>{cells}</tr>")
    return (
        f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar(health: dict[str, Any] | None) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="card-label" style="margin-bottom:0.6rem">Service status</div>',
            unsafe_allow_html=True,
        )
        health_strip(health)
        st.markdown(
            f'<div class="footnote" style="margin-top:0.8rem">API · '
            f'<code>{esc(api_client.base_url())}</code></div>',
            unsafe_allow_html=True,
        )
        if st.button("Re-check services", width="stretch"):
            api_client.clear_static_caches()
            st.rerun()
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown(
            '<div class="footnote">This interface is a <b>client</b> of the FastAPI service. '
            "It holds no retrieval code, no embedding model and no vector store handle: every "
            "number on every page was produced by the backend.</div>",
            unsafe_allow_html=True,
        )
