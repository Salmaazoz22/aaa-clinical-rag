# -*- coding: utf-8 -*-
"""Safety & Abstention.

The examples on this page are *run*, not described. Each button sends its
question to `POST /v1/answer` and renders what the API actually returned —
which gate fired, which signals matched, what the similarity scores were. The
two refusal gates need no language model, so this page is fully functional even
when no LLM key is configured.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as ui
from ui.api_client import ApiError

STATE = "safety_demo_result"

EXAMPLES = [
    {
        "key": "general",
        "title": "General guideline question",
        "question": "At what maximum diameter is elective repair recommended for a man with an asymptomatic fusiform abdominal aortic aneurysm?",
        "outcome": "Answered — if the evidence supports it",
        "tone": "good",
        "why": (
            "Nothing about this question is patient-specific: it asks what the guidelines state "
            "for a described population. It passes the safety gate, and whether it is answered "
            "then depends entirely on whether retrieved evidence clears the floor. "
            "Reaching the model requires an LLM key."
        ),
    },
    {
        "key": "patient",
        "title": "Patient-specific treatment question",
        "question": "My patient is a 72-year-old man with a 5.2 cm aneurysm. Should I operate on him?",
        "outcome": "Refused before retrieval reaches a model",
        "tone": "warn",
        "why": (
            "Three signals fire at once: an explicit patient reference, an individual's "
            "demographics, and an individual-directed ask. The refusal is built locally, so the "
            "patient's details are never sent to a third-party API — and the refusal deliberately "
            "carries no citations, because offering guideline text as an answer to a question "
            "about an individual is the thing the gate exists to prevent."
        ),
    },
    {
        "key": "dosing",
        "title": "Dosing question about an individual",
        "question": "What dose of statin should I start my patient on before his aneurysm repair?",
        "outcome": "Refused — dosing request plus individual-directed ask",
        "tone": "warn",
        "why": (
            "A dosing question with no individual attached is <b>not</b> blocked — "
            "\"what statin dose do the guidelines recommend?\" is a general question about the "
            "corpus and gets answered, or refused for lack of evidence. Refusing it on safety "
            "grounds would hide the fact that the corpus carries no dosing guidance. It is the "
            "combination with an individual that blocks."
        ),
    },
    {
        "key": "unrelated",
        "title": "Unrelated question",
        "question": "What is the best recipe for sourdough bread?",
        "outcome": "Refused by the evidence threshold",
        "tone": "warn",
        "why": (
            "Retrieval still returns its nearest neighbours — a dense retriever always returns "
            "something. None of them clears the similarity floor, so the system refuses and names "
            "the passages it examined and rejected. This is the failure mode most RAG demos hide."
        ),
    },
    {
        "key": "out_of_corpus",
        "title": "Clinical, but outside the corpus",
        "question": "What is the recommended insulin dose for type 2 diabetes?",
        "outcome": "Refused by the evidence threshold",
        "tone": "warn",
        "why": (
            "A real clinical question, plausibly worded, on a topic this corpus does not cover. "
            "Scores land higher than for an unrelated question but still below the floor. This is "
            "the case a similarity threshold earns its keep on."
        ),
    },
]

PRINCIPLES = [
    ("Refusal is a first-class outcome",
     "A refusal is a successful response with its own reason code and gate, not an error and not "
     "a failure. The UI styles it distinctly from an error for exactly that reason."),
    ("A refusal says what it looked at",
     "Every threshold refusal names the passages it examined, their documents and their "
     "similarity scores, so a reader can confirm the gap is real rather than take the refusal on "
     "trust. Those citations are built from the retriever's own records and cannot be fabricated."),
    ("Two layers, neither sufficient alone",
     "The pattern gate catches the unambiguous cases before the model is called. The system "
     "prompt instructs the model to refuse the same class of request, which catches paraphrases "
     "the gate misses. This is a conservative gate, not a classifier, and it will miss things."),
    ("Confidence is a label, never a number",
     "Four fixed values: High, Medium, Low, Insufficient Evidence. No percentage is emitted, "
     "because there is no calculation behind one — a percentage would be false precision about "
     "how well a dense retriever's top-5 supports a clinical claim."),
    ("The disclaimer is attached to everything",
     "Including refusals. It is normalised rather than trusted: if the model returns a different "
     "version it is replaced, and the substitution is recorded as a validator finding."),
]


def render(health: dict[str, Any] | None) -> None:
    st.markdown("# Safety & Abstention")
    st.markdown(
        '<p class="footnote" style="font-size:0.95rem;max-width:72ch">This system does not answer '
        "every question it is asked. Two gates can stop a question before it reaches a model, and "
        "the model itself can decline after reading the evidence. Every example below is executed "
        "live against the API — the results are real, not illustrations.</p>",
        unsafe_allow_html=True,
    )

    llm = (health or {}).get("llm") or {}
    if not llm.get("configured"):
        ui.notice(
            "info",
            "Both refusal gates work without an LLM key",
            "<p class='footnote'>The safety gate and the evidence threshold refuse before any "
            "model call, so every refusal example on this page runs fully right now. The "
            "general-question example will reach the generation step and report that live "
            "generation is unavailable — which is itself the correct behaviour: no answer is "
            "invented in its place.</p>",
        )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## Run the gates")

    st.session_state.setdefault(STATE, {})

    for example in EXAMPLES:
        _example(example)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## The rules behind it")
    cols = st.columns(2, gap="large")
    for index, (title, body) in enumerate(PRINCIPLES):
        with cols[index % 2]:
            st.markdown(
                f'<div class="card"><div class="card-label">{ui.esc(title)}</div>'
                f'<div class="footnote" style="font-size:0.85rem;line-height:1.65">{body}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    ui.notice(
        "warn",
        "What this is not",
        "<p>This is an evidence-retrieval prototype over four abdominal aortic aneurysm "
        "guidelines. It is not clinically validated, it does not provide patient-specific "
        "diagnosis, treatment or dosing, and it must not be used to make clinical decisions. "
        "The safety gate is a conservative pattern matcher, not a classifier: it catches the "
        "unambiguous cases and will miss paraphrases.</p>",
    )


def _example(example: dict[str, Any]) -> None:
    key = example["key"]
    tone_cls = {"good": "ok", "warn": "warn", "bad": "bad"}[example["tone"]]

    st.markdown(
        f'<div class="card tight" style="margin-bottom:0.5rem">'
        f'<div class="ev-head"><span class="badge {tone_cls}">{ui.esc(example["outcome"])}</span>'
        f'<span class="ev-doc">{ui.esc(example["title"])}</span></div>'
        f'<div class="footnote" style="margin-top:0.6rem;font-family:Georgia,serif;'
        f'font-size:0.92rem;color:#13171E">“{ui.esc(example["question"])}”</div>'
        f'<div class="footnote" style="margin-top:0.6rem">{example["why"]}</div></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 4])
    if cols[0].button("Run this", key=f"safety_{key}", width="stretch"):
        with st.spinner("Calling POST /v1/answer…"):
            try:
                st.session_state[STATE][key] = {"result": api_client.answer(example["question"])}
            except ApiError as error:
                st.session_state[STATE][key] = {"error": error}

    stored = st.session_state[STATE].get(key)
    if stored:
        if "error" in stored:
            _error_outcome(stored["error"])
        else:
            _result_outcome(stored["result"])

    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)


def _error_outcome(error: ApiError) -> None:
    if error.status_code == 503 and "API key" in str(error.detail or ""):
        ui.notice(
            "warn",
            "Reached the generation step — live generation unavailable",
            f"<p class='footnote'>{ui.esc(error.detail)}</p>"
            "<p class='footnote'><b>This is the correct outcome</b>: the question passed both "
            "gates, so an answer would have to come from the model. With no key, the API reports "
            "the failure. It does not return a placeholder, a cached answer, or ungrounded text.</p>",
        )
    elif error.status_code == 502:
        ui.notice(
            "bad",
            "The model did not return a usable answer",
            f"<p class='footnote'>{ui.esc(error.detail or error.message)}</p>"
            "<p class='footnote'>No answer is shown, because there is no answer.</p>",
        )
    else:
        ui.backend_unavailable(error)


def _result_outcome(result: dict[str, Any]) -> None:
    safety = result.get("safety") or {}
    retrieval = result.get("retrieval") or {}
    refusal = result.get("refusal") or {}
    refused = bool(result.get("refused"))
    hits = retrieval.get("hits") or []
    top1 = hits[0].get("similarity_score") if hits else None
    floor = (result.get("settings") or {}).get("score_threshold")

    tiles = [
        ui.stat("Outcome", "REFUSED" if refused else "ANSWERED",
                str(refusal.get("gate") or "") if refused else "reached the model",
                "amber" if refused else "green"),
        ui.stat("Safety gate", "BLOCKED" if safety.get("blocked") else "passed",
                f"rule {safety.get('rule')}" if safety.get("rule") else "no rule fired",
                "amber" if safety.get("blocked") else "green"),
        ui.stat("Top-1 similarity", f"{float(top1):.4f}" if top1 is not None else "—",
                f"floor = {floor}",
                "green" if (top1 is not None and floor is not None and top1 >= floor) else "amber"),
        ui.stat("Chunks used", f"{retrieval.get('n_used', 0)}/{retrieval.get('n_retrieved', 0)}",
                f"{retrieval.get('n_dropped_below_threshold', 0)} withheld"),
    ]
    ui.stat_row(tiles)

    signals = safety.get("signals") or []
    if signals:
        chips = "".join(
            f'<span class="check {"n" if safety.get("blocked") else "o"}">{ui.esc(s)}</span>'
            for s in signals
        )
        st.markdown(
            f'<div class="card tight" style="margin-top:0.7rem">'
            f'<div class="card-label">Safety signals matched</div>'
            f'<div class="checks">{chips}</div>'
            + (f'<div class="footnote" style="margin-top:0.6rem">'
               f'{ui.esc(safety.get("detail"))}</div>' if safety.get("detail") else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("What the API actually returned", expanded=False):
        answer = result.get("answer") or {}
        recommendation = str(answer.get("recommendation") or "").strip()
        if recommendation:
            st.markdown(
                f'<div class="answer{" refused" if refused else ""}">'
                f'<div class="body"><p>{ui.esc(recommendation)}</p></div></div>',
                unsafe_allow_html=True,
            )
        ui.pipeline_trace(result)
        if hits:
            ui.score_bars(result)
