# -*- coding: utf-8 -*-
"""The question / answer experience."""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

from ui import api_client, components as ui
from ui.api_client import ApiError
from ui.demo_questions import DEMOS

PLACEHOLDER = "Ask a question about abdominal aortic aneurysm guidelines…"

STATE_QUESTION = "question_text"
STATE_RESULT = "answer_result"
STATE_ASKED = "asked_question"
STATE_ERROR = "answer_error"


def _reset() -> None:
    st.session_state[STATE_QUESTION] = ""
    for key in (STATE_RESULT, STATE_ASKED, STATE_ERROR):
        st.session_state.pop(key, None)


def _run(question: str, top_k: int | None, threshold: float | None) -> None:
    """Call the API once. A refusal is a result; a failure is an error."""
    st.session_state.pop(STATE_RESULT, None)
    st.session_state.pop(STATE_ERROR, None)
    try:
        st.session_state[STATE_RESULT] = api_client.answer(
            question, top_k=top_k, threshold=threshold
        )
        st.session_state[STATE_ASKED] = question
    except ApiError as error:
        st.session_state[STATE_ERROR] = error
        st.session_state[STATE_ASKED] = question


def render(health: dict[str, Any] | None) -> None:
    ui.masthead()

    meta = corpus = None
    meta_error: ApiError | None = None
    try:
        meta = api_client.meta()
        corpus = api_client.corpus()
    except ApiError as error:
        meta_error = error

    ui.system_badges(meta, corpus)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if meta_error is not None:
        ui.backend_unavailable(meta_error)
        return

    llm_down = ui.llm_unavailable_notice(health)

    # -- input ------------------------------------------------------------
    st.session_state.setdefault(STATE_QUESTION, "")
    st.markdown("### Ask a clinical guideline question")
    st.markdown(
        '<div class="footnote" style="margin-bottom:0.6rem">Ask what the guidelines '
        "<i>state</i>. Questions about a specific individual are refused by design — try one "
        "and watch the safety gate fire.</div>",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Question",
        key=STATE_QUESTION,
        placeholder=PLACEHOLDER,
        height=100,
        label_visibility="collapsed",
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        b1, b2, _ = st.columns([1, 1, 2])
        ask = b1.button("Ask", type="primary", width="stretch")
        clear = b2.button("Clear", width="stretch")
    with right:
        with st.expander("Retrieval settings for this question", expanded=False):
            gen = meta.get("generation") or {}
            default_k = int(gen.get("top_k") or 5)
            default_t = float(gen.get("score_threshold") or 0.75)
            st.markdown(
                '<div class="footnote">These are the two request parameters the API accepts. '
                "They change how many chunks are requested and how strict the evidence floor is. "
                "They do <b>not</b> change the retriever, the embedding model, the ranking, or "
                "anything the frozen evaluation measured.</div>",
                unsafe_allow_html=True,
            )
            top_k = st.slider("top_k — chunks requested", 1, 10, default_k)
            threshold = st.slider("evidence floor — minimum similarity", 0.0, 1.0, default_t, 0.01)
            if abs(threshold - default_t) > 1e-9 or top_k != default_k:
                st.markdown(
                    f'<div class="footnote" style="color:#9A6511">Overriding the configured '
                    f"defaults (top_k={default_k}, floor={default_t}).</div>",
                    unsafe_allow_html=True,
                )

    if clear:
        _reset()
        st.rerun()

    # -- demo questions ---------------------------------------------------
    with st.expander("Demo questions — each one exercises a different capability", expanded=False):
        st.markdown(
            '<div class="footnote" style="margin-bottom:0.8rem">Guideline questions are taken '
            "verbatim from the frozen evaluation set the published retrieval metrics were "
            "measured on. Nothing below states an expected answer — only what the pipeline does "
            "with the question.</div>",
            unsafe_allow_html=True,
        )
        for index, demo in enumerate(DEMOS):
            cols = st.columns([1.05, 3], gap="medium")
            with cols[0]:
                if st.button(demo.label, key=f"demo_{index}", width="stretch"):
                    st.session_state[STATE_QUESTION] = demo.question
                    _run(demo.question, top_k, threshold)
                    st.rerun()
            with cols[1]:
                tone = "warn" if demo.kind == "refuse" else "accent"
                st.markdown(
                    f'<div class="badge {tone}">{ui.esc(demo.capability)}</div>'
                    f'<div class="footnote" style="margin-top:0.35rem">'
                    f'<b>{ui.esc(demo.question)}</b><br>{ui.esc(demo.expects)}</div>',
                    unsafe_allow_html=True,
                )
            if index < len(DEMOS) - 1:
                st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)

    if ask:
        if not question.strip():
            ui.notice(
                "warn",
                "Enter a question first",
                "<p class='footnote'>An empty or whitespace-only question is rejected by the API "
                "rather than sent to retrieval.</p>",
            )
        else:
            with st.spinner(
                "Running the pipeline — safety screening → retrieval → evidence threshold → "
                "generation → citation validation"
            ):
                _run(question.strip(), top_k, threshold)

    # -- output -----------------------------------------------------------
    error: ApiError | None = st.session_state.get(STATE_ERROR)
    result: dict[str, Any] | None = st.session_state.get(STATE_RESULT)

    if error is not None:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        _render_error(error, llm_down)
        return

    if result is None:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        ui.notice(
            "info",
            "No question asked yet",
            "<p class='footnote'>Ask a question above, or pick one from the demo list. Every "
            "answer arrives with the passages it was built from, the similarity score of each "
            "one, and the validator's verdict on every citation.</p>",
        )
        return

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown(f"### {ui.esc(st.session_state.get(STATE_ASKED, ''))}", unsafe_allow_html=True)

    ui.answer_panel(result)
    ui.pipeline_trace(result)

    tabs = st.tabs([
        "Retrieved evidence",
        "Evidence → answer trace",
        "Retrieval scores",
        "Validator findings",
        "Provenance",
        "Raw response",
    ])

    with tabs[0]:
        ui.evidence_panel(result)
    with tabs[1]:
        st.markdown("#### Evidence → Answer Trace")
        st.markdown(
            '<div class="footnote" style="margin-bottom:1rem">Claim → citation → chunk → '
            "guideline → page, for every citation the model emitted, checked against the "
            "chunks that were actually sent to it.</div>",
            unsafe_allow_html=True,
        )
        ui.citation_trace(result)
    with tabs[2]:
        ui.score_bars(result)
    with tabs[3]:
        st.markdown("#### Citation validator findings")
        ui.validation_findings(result)
    with tabs[4]:
        ui.provenance_panel(meta, result)
    with tabs[5]:
        _raw_response(result)


def _render_error(error: ApiError, llm_down: bool) -> None:
    """Explain a failed request without ever standing in an answer's place."""
    if error.status_code == 502:
        ui.notice(
            "bad",
            "Live generation could not be completed",
            f"<p>{ui.esc(error.detail or error.message)}</p>"
            "<p>The retrieval and safety layers ran; the language model did not return a usable "
            "answer. <b>No answer is shown, because there is no answer</b> — this system never "
            "substitutes a placeholder, a cached response, or ungrounded text.</p>",
        )
    elif error.status_code == 503 and "API key" in str(error.detail or ""):
        ui.notice(
            "warn",
            "Live generation unavailable — no LLM API key",
            f"<p>{ui.esc(error.detail)}</p>"
            "<p>Retrieval, the safety gate and the evidence threshold all work without a key. "
            "Only questions that pass every gate and reach the model need one.</p>",
        )
    elif error.status_code == 400:
        ui.notice(
            "warn",
            "The question was rejected",
            f"<p>{ui.esc(error.detail or error.message)}</p>",
        )
    else:
        ui.backend_unavailable(error)


def _raw_response(result: dict[str, Any]) -> None:
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    answer = result.get("answer") or {}

    col_a, col_b = st.columns(2)
    col_a.download_button(
        "Download full audit record (JSON)",
        data=payload,
        file_name="clinical_rag_response.json",
        mime="application/json",
        width="stretch",
    )
    col_b.download_button(
        "Download answer text (.txt)",
        data=_answer_text(result),
        file_name="clinical_rag_answer.txt",
        mime="text/plain",
        width="stretch",
    )

    st.markdown("##### Answer text — select and copy")
    st.code(_answer_text(result), language="text")

    st.markdown("##### Full API response")
    st.markdown(
        '<div class="footnote" style="margin-bottom:0.5rem">This is the complete, unmodified '
        "<code>GET</code> payload from <code>POST /v1/answer</code>. Everything on this page is "
        "rendered from it — nothing is added, and nothing is filtered out.</div>",
        unsafe_allow_html=True,
    )
    st.json(result, expanded=False)

    if answer:
        st.markdown("##### Structured answer object")
        st.json(answer, expanded=False)


def _answer_text(result: dict[str, Any]) -> str:
    """A plain-text rendering that keeps the citations attached to the prose."""
    answer = result.get("answer") or {}
    lines = [
        f"QUESTION: {result.get('query', '')}",
        "",
        "RECOMMENDATION:",
        str(answer.get("recommendation") or "").strip(),
        "",
        f"CONFIDENCE: {answer.get('confidence', '—')}",
    ]

    if result.get("refused"):
        refusal = result.get("refusal") or {}
        lines += ["", f"REFUSED: reason={refusal.get('reason')} gate={refusal.get('gate')}"]

    supporting = [e for e in (answer.get("supporting_evidence") or []) if isinstance(e, dict)]
    if supporting:
        lines += ["", "SUPPORTING EVIDENCE:"]
        lines += [f"  - {e.get('claim')}  [{e.get('chunk_id')}]" for e in supporting]

    citations = [c for c in (answer.get("citations") or []) if isinstance(c, dict)]
    if citations:
        lines += ["", "CITATIONS:"]
        for index, citation in enumerate(citations, start=1):
            lines.append(
                f"  [{index}] {citation.get('document')} — {citation.get('section')} — "
                f"page {citation.get('page')} — {citation.get('chunk_id')} "
                f"(score {citation.get('retrieval_score')})"
            )
            if citation.get("excerpt"):
                lines.append(f"      \"{citation['excerpt']}\"")

    validation = result.get("validation") or {}
    lines += [
        "",
        f"CITATION VALIDATION: {'PASS' if validation.get('ok') else 'FAIL'} "
        f"({validation.get('n_errors', 0)} errors, {validation.get('n_warnings', 0)} warnings)",
    ]

    retrieval = result.get("retrieval") or {}
    lines.append(
        f"RETRIEVAL: {retrieval.get('n_retrieved', 0)} retrieved, "
        f"{retrieval.get('n_used', 0)} used as evidence, "
        f"{retrieval.get('n_dropped_below_threshold', 0)} below the floor "
        f"({(result.get('settings') or {}).get('score_threshold')})"
    )

    if answer.get("disclaimer"):
        lines += ["", "DISCLAIMER:", str(answer["disclaimer"])]

    return "\n".join(lines)
