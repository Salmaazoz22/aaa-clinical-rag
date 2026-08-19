# -*- coding: utf-8 -*-
"""The Ask page.

The screen reads Question → Reasoning → Evidence → Answer → Citations without
anyone explaining it. Evidence is a permanent column, never an expander at the
bottom: the whole claim of this system is that an answer arrives with the
passages it was built from, and burying them would contradict it.

Everything rendered here maps to a field in the API response. The derivations
that are not one-to-one are documented at their point of use and listed in
INVENTORY.md §3.
"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

from ui import api_client, components as c
from ui.api_client import ApiError
from ui.demo_questions import DEMOS
from ui.icons import icon
from ui.shell import Context
from ui.pdf_highlighter import render_highlighted_pdf_page
from ui.transcription import transcribe_audio_bytes

Q = "ask_question"
RESULT = "ask_result"
ASKED = "ask_asked"
ERROR = "ask_error"
PENDING = "ask_pending"
EXPANDED = "ask_expanded"

PLACEHOLDER = "At what diameter do the guidelines recommend elective repair in men?"


# ---------------------------------------------------------------------------
# Derivations — each one documented, none invented
# ---------------------------------------------------------------------------

def _citation_states(result: dict[str, Any]) -> list[str]:
    """One of ok | warn | bad per citation.

    Derived from two real fields, never guessed:
      * `citations_resolved[i].resolved` — did the cited chunk actually reach
        the model? False means the id was fabricated.
      * validator findings whose `location` starts `citations[i]` — the API
        stamps the citation index into the location, so attribution is exact.

    An error-severity finding, or an unresolved chunk, is `bad`. A
    warning-severity finding is `warn`. Clean is `ok`.
    """
    citations = result.get("answer", {}).get("citations") or []
    resolved = result.get("citations_resolved") or []
    findings = (result.get("validation") or {}).get("findings") or []

    by_index: dict[int, list[str]] = {}
    for finding in findings:
        location = str(finding.get("location") or "")
        if not location.startswith("citations["):
            continue
        try:
            i = int(location[len("citations["):location.index("]")])
        except (ValueError, IndexError):
            continue
        by_index.setdefault(i, []).append(str(finding.get("severity")))

    states = []
    for i in range(len(citations)):
        record = resolved[i] if i < len(resolved) else {}
        severities = by_index.get(i, [])
        if not record.get("resolved") or "error" in severities:
            states.append("bad")
        elif severities:
            states.append("warn")
        else:
            states.append("ok")
    return states


def _verdict(result: dict[str, Any], states: list[str]) -> tuple[str, int]:
    """The grounding verdict word and the count of fully-clean citations."""
    if result.get("refused"):
        return "Abstained", 0
    verified = sum(1 for s in states if s == "ok")
    if not states:
        return "Ungrounded", 0
    if verified == len(states):
        return "Grounded", verified
    return "Partially grounded", verified


def _cited_index(chunk_id: str, result: dict[str, Any]) -> int | None:
    """Which citation number this chunk backs, if any. 1-based."""
    for i, citation in enumerate(result.get("answer", {}).get("citations") or []):
        if str(citation.get("chunk_id")) == chunk_id:
            return i + 1
    return None


def _stages(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Real outcomes, read back out of the response.

    The API answers in ONE call and reports no intermediate milestones, so this
    is a record of what happened — never an animation. While a request is in
    flight the caller uses `_pending_stages()` instead, which marks everything
    it cannot yet observe as pending.
    """
    safety = result.get("safety") or {}
    retrieval = result.get("retrieval") or {}
    validation = result.get("validation") or {}
    settings = result.get("settings") or {}
    gate = str((result.get("refusal") or {}).get("gate") or "")
    hits = retrieval.get("hits") or []
    floor = settings.get("score_threshold")

    if safety.get("blocked"):
        parse = ("failed", f"blocked · rule {safety.get('rule')}")
    else:
        parse = ("complete", "no patient-specific signal")

    n = retrieval.get("n_retrieved", 0)
    top = f" · top {hits[0]['similarity_score']:.4f}" if hits else ""
    retrieve = ("complete", f"{n} chunks{top}") if n else ("failed", "nothing returned")

    used = retrieval.get("n_used", 0)
    if gate == "threshold":
        ground = ("failed", f"0 of {n} above {floor}")
    elif safety.get("blocked"):
        ground = ("pending", "not reached")
    else:
        ground = ("complete", f"{used} of {n} above {floor}")

    if validation.get("ok"):
        validate = ("complete",
                    f"{validation.get('n_errors', 0)} errors · {validation.get('n_warnings', 0)} warnings")
    else:
        validate = ("failed", f"{validation.get('n_errors', 0)} errors")

    return [("Parse", *parse), ("Retrieve", *retrieve), ("Ground", *ground), ("Validate", *validate)]


def _pending_stages() -> list[tuple[str, str, str]]:
    """In flight. Only the stage we can honestly claim has begun is `active`."""
    return [("Parse", "active", "in flight"), ("Retrieve", "pending", ""),
            ("Ground", "pending", ""), ("Validate", "pending", "")]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _clear() -> None:
    for key in (RESULT, ASKED, ERROR, PENDING, EXPANDED):
        st.session_state.pop(key, None)
    st.session_state[Q] = ""


def _submit(question: str, top_k: int | None, threshold: float | None) -> None:
    st.session_state.pop(RESULT, None)
    st.session_state.pop(ERROR, None)
    st.session_state[ASKED] = question
    try:
        st.session_state[RESULT] = api_client.answer(question, top_k=top_k, threshold=threshold)
    except ApiError as error:
        st.session_state[ERROR] = error


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def _composer(ctx: Context) -> None:
    st.session_state.setdefault(Q, "")

    # Check if an example pill was selected in previous rerun before text_area is instantiated
    picked_demo = st.session_state.get("ask_demo")
    if picked_demo:
        demo = next((d for d in DEMOS if d.label == picked_demo), None)
        if demo:
            st.session_state[Q] = demo.question
        st.session_state["ask_demo"] = None

    with st.container(key="composer"):
        c.write('<div class="eyebrow">Clinical question</div>')

        # Voice input popover (Speech-to-Text)
        voice_popover = st.popover("Voice Input / Speech-to-Text", width="stretch")
        with voice_popover:
            c.write(
                '<div class="eyebrow">Record Voice Question</div>'
                '<div class="tiny" style="margin:4px 0 8px">1. Record your clinical question.<br>'
                '2. Click <b>Convert Voice to Text</b> below.<br>'
                '3. Verify/edit the text in the input box before submitting.</div>'
            )
            audio_file = st.audio_input("Record clinical question", key="ask_voice_input")
            if audio_file is not None:
                if st.button("Convert Voice to Text", type="primary", key="btn_transcribe_voice", width="stretch"):
                    audio_bytes = audio_file.read()
                    if audio_bytes:
                        with st.spinner("Transcribing audio..."):
                            txt, err = transcribe_audio_bytes(
                                audio_bytes, filename=getattr(audio_file, "name", "voice.wav")
                            )
                            if txt:
                                st.session_state[Q] = txt
                                st.success("Transcribed! Verify or edit the question below before clicking Ask.")
                            elif err:
                                st.warning(err)
                    else:
                        st.warning("Please record audio before converting.")

        st.text_area("Question", key=Q, placeholder=PLACEHOLDER, height=96,
                     label_visibility="collapsed")

        chip_col, ctrl_col = st.columns([1.9, 2], gap="medium")

        with chip_col:
            labels = [d.label for d in DEMOS]
            st.pills("Examples", labels, key="ask_demo", label_visibility="collapsed")

        with ctrl_col:
            pop, ask_col, clr_col = st.columns([1.5, 1, 1], gap="small")
            with pop:
                with st.popover("Retrieval", width="stretch"):
                    c.write('<div class="eyebrow">Request parameters</div>'
                            '<div class="tiny" style="margin:6px 0 12px">These are the two '
                            'parameters the API accepts. They change how many chunks are requested '
                            'and how strict the evidence floor is. They do not change the '
                            'retriever, the embedding model, or the ranking.</div>')
                    st.slider("top_k — chunks requested", 1, 10, ctx.top_k, key="ask_top_k")
                    st.slider("evidence floor", 0.0, 1.0, ctx.threshold, 0.01, key="ask_threshold")
            submitted = ask_col.button("Ask", type="primary", width="stretch", key="ask_go")
            cleared = clr_col.button("Clear", width="stretch", key="ask_clear")

    top_k = st.session_state.get("ask_top_k", ctx.top_k)
    threshold = st.session_state.get("ask_threshold", ctx.threshold)

    # Idle caliper — a calibrated instrument waiting for a reading.
    c.write(f'<div class="panel" style="margin-top:-8px;padding-top:8px">'
            f'{c.caliper([], threshold, "idle", show_legend=True)}</div>')

    if cleared:
        _clear()
        st.rerun()

    if submitted:
        question = str(st.session_state.get(Q) or "").strip()
        if not question:
            c.write(c.error_state(
                "Enter a question first",
                "<p>An empty question is rejected by the API before retrieval runs. Type one above, "
                "or pick an example.</p>", glyph="alert"))
        else:
            with st.spinner(""):
                st.session_state[PENDING] = True
                _submit(question, top_k, threshold)
                st.session_state.pop(PENDING, None)
            st.rerun()


def _query_head(question: str) -> None:
    c.write(
        f'<div class="query-head"><div class="q">{c.esc(question)}</div>'
        f'<span class="mono tiny" style="white-space:nowrap">{icon("edit", 13)} edit above</span></div>'
    )


# ---------------------------------------------------------------------------
# Evidence rail
# ---------------------------------------------------------------------------

def _passage(chunk_id: str, hit: dict[str, Any]) -> tuple[str, str]:
    """Full stored text where the API can supply it; otherwise the preview, labelled.

    `retrieval.hits[].text_preview` is truncated to 240 characters by the
    pipeline, so full evidence needs the documented audit lookup.
    """
    try:
        payload = api_client.chunk(chunk_id)
    except ApiError:
        preview = str(hit.get("text_preview") or "").strip()
        return (preview + " …" if preview else ""), "Preview only — full text unavailable"
    text = str(payload.get("chunk_text") or "").strip()
    if not text:
        return str(hit.get("text_preview") or ""), "Preview only — no stored text returned"
    return text, f"Full indexed text · GET /v1/chunks/{chunk_id}"


def _evidence_rail(result: dict[str, Any], ctx: Context) -> None:
    retrieval = result.get("retrieval") or {}
    hits = retrieval.get("hits") or []
    threshold = float((result.get("settings") or {}).get("score_threshold") or ctx.threshold)
    used = set(retrieval.get("used_chunk_ids") or [])
    years = ctx.years

    scores = [h.get("similarity_score", 0.0) for h in hits]
    labels = [f"{h.get('document_id')} {h.get('section') or ''}".strip() for h in hits]

    c.write(
        '<div class="rail-head"><div class="row">'
        '<span class="eyebrow">Evidence</span>'
        f'<span class="mono tiny">{len(hits)} retrieved · {len(used)} above floor</span></div>'
        f"{c.caliper(scores, threshold, 'at-rest', labels=labels)}</div>"
    )

    if not hits:
        c.write(c.empty_state(
            "No passages returned",
            "<p>The vector store returned nothing for this question, so there was nothing to "
            "ground an answer in.</p>", glyph="alert"))
        return

    expanded: set[str] = st.session_state.setdefault(EXPANDED, set())

    for i, hit in enumerate(hits, start=1):
        chunk_id = str(hit.get("chunk_id"))
        above = chunk_id in used
        passage, note = _passage(chunk_id, hit)
        is_open = chunk_id in expanded
        c.write(c.evidence_card(
            hit, i,
            above_threshold=above,
            cited_as=_cited_index(chunk_id, result),
            year=years.get(str(hit.get("document_id"))),
            passage=passage, passage_note=note, expanded=is_open,
        ))

        # Render PDF source page preview with highlighted bounding box
        doc_id = str(hit.get("document_id") or hit.get("source_file") or "")
        page_num = hit.get("page_number") or hit.get("page_start") or 1
        if isinstance(page_num, str) and page_num.isdigit():
            page_num = int(page_num)
        elif not isinstance(page_num, int):
            page_num = 1

        with st.expander(f"View Source PDF Page {page_num} (Highlight)", expanded=False):
            img_bytes = render_highlighted_pdf_page(doc_id, page_number=page_num, excerpt=passage)
            if img_bytes:
                st.image(
                    img_bytes,
                    caption=f"Source PDF Page {page_num} ({doc_id}) — Highlighted Evidence",
                    use_container_width=True,
                )
            else:
                c.write('<div class="tiny" style="color:var(--text-muted)">PDF page preview unavailable.</div>')

        if len(passage) > 320:
            label = "Collapse passage" if is_open else "Expand passage"
            if st.button(label, key=f"exp-{chunk_id}", width="stretch"):
                expanded.symmetric_difference_update({chunk_id})
                st.rerun()


# ---------------------------------------------------------------------------
# Answer column
# ---------------------------------------------------------------------------

_REFUSAL_COPY = {
    "patient_specific_request": (
        "Refused by design",
        "This question asks for a decision about a specific individual. It was refused before "
        "retrieval reached a language model, so no patient detail was sent to a third-party API. "
        "The refusal deliberately carries no citations: offering guideline text as an answer to a "
        "question about a person is exactly what this gate exists to prevent.",
        "A general form of the same question can be answered — what the guidelines recommend for "
        "the relevant population, diameter band, or repair modality. The individual decision "
        "belongs with the treating clinician.",
    ),
    "all_scores_below_threshold": (
        "No evidence above threshold",
        "Retrieval ran and returned passages, but none of them cleared the evidence floor. The "
        "system will not build a clinical statement on passages that are only topically adjacent, "
        "so it refused instead of answering. The caliper on the right shows every score against "
        "the line.",
        "This index covers abdominal aortic aneurysm screening, surveillance, medical management "
        "and repair, across ESVS 2024, NICE NG156, SVS 2018 and USPSTF 2019. A question outside "
        "that scope cannot be answered from it.",
    ),
    "no_chunks_retrieved": (
        "Nothing retrieved",
        "Nothing in the indexed guideline corpus was close enough to this question to return as "
        "evidence at all.",
        "Try a question about AAA screening, surveillance, medical management or repair.",
    ),
    "evidence_not_specific_enough": (
        "Evidence not specific enough",
        "Evidence cleared the floor and was sent to the model, and the model judged it topically "
        "related but not specific enough to answer the question as asked. That judgement needs the "
        "evidence read, so it cannot be made by a threshold.",
        "A more specific question — naming the population, the diameter band, or the procedure — "
        "usually retrieves a passage that addresses it directly.",
    ),
    "potential_emergency_presentation": (
        "Potential emergency — seek immediate care",
        "This query describes symptoms that may indicate a medical emergency. "
        "This system cannot assess whether this is an emergency and must not be used in place of "
        "emergency medical evaluation. No guideline text has been cited here — doing so would "
        "look like clinical guidance for an acute individual situation, which this system is "
        "not designed or validated to provide.",
        "Call emergency services immediately (999 / 112 / 911) or go to the nearest emergency "
        "department now if you or someone else is experiencing sudden severe abdominal or back "
        "pain, collapse, or other signs of a possible vascular emergency. Do not wait for an "
        "online response.",
    ),
}


def _answer_column(result: dict[str, Any]) -> None:
    c.write(c.stage_tracker(_stages(result)))

    answer = result.get("answer") or {}
    refusal = result.get("refusal") or {}

    if result.get("refused"):
        reason = str(refusal.get("reason") or "")
        heading, explanation, answerable = _REFUSAL_COPY.get(
            reason,
            ("Refused", "The system declined to answer this question from the retrieved evidence.", ""),
        )
        safety = result.get("safety") or {}
        c.write(c.abstention_panel(
            heading=heading,
            rule=safety.get("rule") if safety.get("blocked") else None,
            reason=reason,
            explanation=explanation,
            signals=safety.get("signals") or [],
            answerable=answerable,
        ))
        c.write(f'<div class="panel" style="margin-top:12px">'
                f'<div class="eyebrow" style="margin-bottom:10px">What the system reported</div>'
                f'<div class="serif" style="font-size:.95rem;line-height:1.6">'
                f'{c.esc(answer.get("recommendation"))}</div></div>')
        return

    completion = (result.get("generation") or {}).get("completion") or {}
    states = _citation_states(result)
    verdict, verified = _verdict(result, states)

    c.write(c.answer_panel(
        answer.get("recommendation") or "",
        n_citations=len(answer.get("citations") or []),
        model=completion.get("model"),
        latency=completion.get("latency_s"),
        verdict=verdict, segments=states, verified=verified,
    ))

    conflicts = [x for x in (answer.get("evidence_conflicts") or []) if isinstance(x, dict)]
    for conflict in conflicts:
        positions = "".join(
            f'<div style="margin-top:8px"><div class="mono tiny">'
            f'{c.esc(", ".join(p.get("chunk_ids") or []))}</div>'
            f'<div class="serif" style="font-size:.95rem;line-height:1.55">{c.esc(p.get("position"))}</div></div>'
            for p in (conflict.get("positions") or []) if isinstance(p, dict)
        )
        c.write(f'<div class="conflict" style="margin-top:12px">'
                f'<div class="eyebrow">Guidelines disagree</div>'
                f'<div style="font-weight:600;margin-top:6px">{c.esc(conflict.get("topic"))}</div>'
                f"{positions}</div>")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

def _empty_right(ctx: Context) -> None:
    """Demonstrates rather than describes: the real corpus, from the API."""
    docs = (ctx.corpus or {}).get("documents") or []
    total = (ctx.corpus or {}).get("total_indexed_chunks")

    rows = []
    for doc in docs:
        chunks = doc.get("indexed_chunks")
        count = f"{chunks:,}" if isinstance(chunks, int) else "—"
        rows.append(
            '<div style="display:flex;justify-content:space-between;gap:12px;padding:7px 0;'
            'border-bottom:1px solid var(--line)">'
            f'<span style="font-size:.8125rem">{c.esc(doc.get("source_organization"))}</span>'
            f'<span class="mono tiny">{c.esc(doc.get("publication_year"))} · {count}</span></div>'
        )

    total_txt = f"{total:,}" if isinstance(total, int) else "—"
    note = (
        "Every answer arrives with the passages it was built from, each one&rsquo;s similarity "
        "score against the evidence floor, and the validator&rsquo;s verdict on every citation."
    )
    c.write(
        '<div class="rail-head"><div class="row"><span class="eyebrow">Corpus</span>'
        f'<span class="mono tiny">{total_txt} chunks</span></div></div>'
        f'<div class="panel">{"".join(rows)}'
        f'<div class="tiny" style="margin-top:14px;line-height:1.55">{note}</div></div>'
    )


def _render_error(error: ApiError) -> None:
    if error.status_code == 502:
        c.write(c.error_state(
            "Generation did not complete",
            f"<p>{c.esc(error.detail or error.message)}</p>"
            "<p>Retrieval and the safety gate ran. The model did not return a usable answer, so "
            "there is no answer to show — this system never substitutes a placeholder or a cached "
            "response.</p>", glyph="alert"))
    elif error.status_code == 503 and "API key" in str(error.detail or ""):
        c.write(c.error_state(
            "Live generation unavailable",
            f"<p>{c.esc(error.detail)}</p>"
            "<p>Set the provider key in <code>.env</code> and restart the API. Retrieval and both "
            "refusal gates work without one.</p>", glyph="alert"))
    elif error.status_code == 400:
        c.write(c.error_state("Question rejected",
                              f"<p>{c.esc(error.detail or error.message)}</p>", glyph="alert"))
    elif error.kind == "offline":
        c.write(c.error_state(
            "Backend unavailable",
            f"<p>Nothing is answering at <code>{c.esc(api_client.base_url())}</code>. This page is "
            "a client of the FastAPI service and has no retrieval of its own.</p>"
            "<p>Start it with <code>uvicorn api.main:app --port 8000</code>, then use "
            "<b>Re-check services</b> in the rail.</p>"))
    else:
        c.write(c.error_state("Request failed",
                              f"<p>{c.esc(error.message)}</p>"
                              + (f"<p class='tiny'>{c.esc(error.detail)}</p>" if error.detail else "")))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render(ctx: Context) -> None:
    # A dead backend must be announced, not silently replaced by an empty
    # composer. `load_context` never reaches /v1/meta when /health already
    # failed, so meta_error is None in that case and this check is what carries
    # the offline experience.
    if not ctx.api_up:
        c.write(c.error_state(
            "Backend unavailable",
            f"<p>Nothing is answering at <code>{c.esc(api_client.base_url())}</code>. This page is "
            "a client of the FastAPI service and has no retrieval of its own, so it cannot show "
            "results until the API is up.</p>"
            "<p>Start it from the project root with "
            "<code>uvicorn api.main:app --host 127.0.0.1 --port 8000</code>, then use "
            "<b>Re-check services</b> in the rail.</p>"))
        return

    if ctx.meta_error is not None:
        _render_error(ctx.meta_error)
        return

    result: dict[str, Any] | None = st.session_state.get(RESULT)
    error: ApiError | None = st.session_state.get(ERROR)
    asked: str = st.session_state.get(ASKED, "")

    left, right = st.columns([62, 38], gap="large")

    with left:
        if result is None and error is None:
            _composer(ctx)
        else:
            _query_head(asked)
            with st.expander("Ask another question", expanded=False):
                _composer(ctx)

        if error is not None:
            c.write('<div style="height:12px"></div>')
            _render_error(error)
        elif result is not None:
            _answer_column(result)
        else:
            c.write('<div style="height:12px"></div>')
            c.write(c.empty_state(
                "Ask a question to begin",
                "<p>Ask what the guidelines <i>state</i>. A question about a specific individual is "
                "refused before it reaches a model — try one and watch the gate fire.</p>"))

    with right:
        if result is not None:
            _evidence_rail(result, ctx)
        elif error is not None:
            # Deliberately nothing here. A skeleton in this slot was shimmering
            # for evidence that had already failed to arrive — decoration that
            # actively implies loading when nothing is loading. The error panel
            # in the answer column already says what happened and what to do.
            pass
        else:
            _empty_right(ctx)

    if result is not None:
        _detail_tabs(result)


def _detail_tabs(result: dict[str, Any]) -> None:
    c.write('<hr class="hair">')
    findings_tab, trace_tab, raw_tab = st.tabs(["Validator findings", "Citation trace", "Raw response"])

    with findings_tab:
        findings = (result.get("validation") or {}).get("findings") or []
        if not findings:
            c.write(c.empty_state(
                "No findings",
                "<p>Every citation resolved to a chunk that was actually sent to the model, every "
                "excerpt was found in its cited chunk, and every metadata field matched the "
                "retriever's own record.</p>", glyph="check"))
        for finding in findings:
            tone = "caution" if finding.get("severity") == "error" else "neutral"
            extra = ""
            if finding.get("expected") is not None or finding.get("actual") is not None:
                extra = c.definition_list([
                    ("expected", json.dumps(finding.get("expected"), ensure_ascii=False)),
                    ("actual", json.dumps(finding.get("actual"), ensure_ascii=False)),
                ])
            c.write(
                f'<div class="panel" style="padding:14px 16px;margin-bottom:8px">'
                f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
                f'{c.status_pill(str(finding.get("severity")).upper(), tone)}'
                f'{c.status_pill(str(finding.get("code")), "neutral")}'
                f'{c.status_pill(str(finding.get("location") or "—"), "neutral")}</div>'
                f'<div style="margin-top:10px;font-size:.875rem;line-height:1.55">'
                f'{c.esc(finding.get("message"))}</div>{extra}</div>')

    with trace_tab:
        states = _citation_states(result)
        resolved = result.get("citations_resolved") or []
        citations = result.get("answer", {}).get("citations") or []
        if not citations:
            c.write(c.empty_state("No citations", "<p>This response carries no citations.</p>",
                                  glyph="info"))
        for i, citation in enumerate(citations):
            record = resolved[i] if i < len(resolved) else {}
            tone = {"ok": "verified", "warn": "neutral", "bad": "caution"}[states[i]]
            word = {"ok": "verified", "warn": "verified with warnings",
                    "bad": "not verified"}[states[i]]
            doc_id = str(record.get("document_id") or citation.get("document") or citation.get("source_file") or "")
            page_num = record.get("page_number") or record.get("page_start") or citation.get("page") or 1
            if isinstance(page_num, str) and page_num.isdigit():
                page_num = int(page_num)
            elif not isinstance(page_num, int):
                page_num = 1
            excerpt = str(citation.get("excerpt") or "")

            c.write(
                f'<div class="panel" style="padding:16px;margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;gap:10px">'
                f'{c.citation_chip(i + 1)}{c.status_pill(word, tone)}</div>'
                + c.definition_list([
                    ("Chunk", record.get("chunk_id") or citation.get("chunk_id")),
                    ("Guideline", record.get("document_id") or citation.get("document")),
                    ("Section", record.get("section") or citation.get("section")),
                    ("Page", c._pages(record) if record.get("resolved") else citation.get("page")),
                    ("Similarity", c.num(record.get("retrieval_score"))),
                    ("Retrieval rank", record.get("rank")),
                ])
                + (f'<div class="serif" style="font-size:.95rem;line-height:1.55;margin-top:10px;'
                   f'border-left:2px solid var(--line);padding-left:12px">'
                   f'{c.esc(excerpt)}</div>' if excerpt else "")
                + "</div>")

            with st.expander(f"View Source PDF Page {page_num} (Highlighted Citation [{i+1}])", expanded=False):
                img_bytes = render_highlighted_pdf_page(doc_id, page_number=page_num, excerpt=excerpt)
                if img_bytes:
                    st.image(
                        img_bytes,
                        caption=f"Source PDF Page {page_num} ({doc_id}) — Citation [{i+1}]",
                        use_container_width=True,
                    )
                else:
                    c.write('<div class="tiny" style="color:var(--text-muted)">PDF page preview unavailable.</div>')

    with raw_tab:
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button("Download audit record (JSON)", payload,
                           file_name="clinical_rag_response.json", mime="application/json")
        c.write('<div class="tiny" style="margin:10px 0">The complete, unmodified payload from '
                '<span class="mono">POST /v1/answer</span>. Everything on this page is rendered '
                'from it — nothing added, nothing filtered out.</div>')
        st.json(result, expanded=False)
