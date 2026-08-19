# -*- coding: utf-8 -*-
"""Safety & abstention — a rule table with a live "try it" per rule.

The rules are described on the left and executed on the right. Clicking a rule's
chip loads its triggering question into the Ask composer and runs it, so the
claim is demonstrated rather than asserted. Both refusal gates run before any
model call, so this page is fully functional with no LLM key.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as c
from ui.api_client import ApiError
from ui.shell import Context

STATE = "safety_runs"

RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "B1",
        "rule": "An individual is named as the subject",
        "why": "\"my patient\", \"this patient\", a relative, or a title plus surname. Unambiguous "
               "on its own, so it blocks without needing a second signal.",
        "question": "My patient is a 72-year-old man with a 5.2 cm aneurysm. Should I operate on him?",
        "expect": "safety",
    },
    {
        "id": "B2",
        "rule": "A diagnosis is requested for an individual",
        "why": "\"does he have\", \"is this an aneurysm\", \"what is the diagnosis\". Reporting what "
               "a guideline says is not the same as diagnosing a person.",
        "question": "Does my father have a ruptured aneurysm?",
        "expect": "safety",
    },
    {
        "id": "B3",
        "rule": "An individual's demographics plus an ask about what to do for them",
        "why": "Age in years is the signal, not a measurement — \"45 mm\" and \"aged 65 to 75\" "
               "describe populations, not a person.",
        "question": "A 68-year-old man has a 5.8 cm aneurysm — should we operate?",
        "expect": "safety",
    },
    {
        "id": "B4",
        "rule": "A dosing request plus an individual-directed ask",
        "why": "A dosing question with no individual attached is NOT blocked. \"What statin dose do "
               "the guidelines recommend?\" is a question about the corpus and gets answered, or "
               "refused for lack of evidence. It is the combination that blocks.",
        "question": "What dose of statin should I start my patient on before repair?",
        "expect": "safety",
    },
    {
        "id": "floor",
        "rule": "No passage clears the evidence floor",
        "why": "A dense retriever always returns something. Retrieval runs, returns its nearest "
               "neighbours, and none clears the similarity floor — so the system refuses and names "
               "the passages it examined and rejected.",
        "question": "What is the best recipe for sourdough bread?",
        "expect": "threshold",
    },
    {
        "id": "floor",
        "rule": "Clinical, but outside this corpus",
        "why": "A real clinical question on a topic these four guidelines do not cover. Scores land "
               "higher than for an unrelated question but still below the floor. This is the case a "
               "similarity threshold earns its keep on.",
        "question": "What is the recommended insulin dose for type 2 diabetes?",
        "expect": "threshold",
    },
)


def _run(index: int, question: str) -> None:
    runs = st.session_state.setdefault(STATE, {})
    try:
        runs[index] = {"result": api_client.answer(question)}
    except ApiError as error:
        runs[index] = {"error": error}


def _outcome(entry: dict[str, Any], ctx: Context) -> None:
    if "error" in entry:
        error = entry["error"]
        if error.status_code == 503 and "API key" in str(error.detail or ""):
            c.write(c.error_state(
                "Reached generation — no key configured",
                "<p>The question passed both gates, so an answer would have to come from the model. "
                "The API reports the failure rather than returning a placeholder.</p>", glyph="alert"))
        else:
            c.write(c.error_state("Request failed", f"<p>{c.esc(error.message)}</p>"))
        return

    result = entry["result"]
    safety = result.get("safety") or {}
    retrieval = result.get("retrieval") or {}
    refusal = result.get("refusal") or {}
    hits = retrieval.get("hits") or []
    threshold = float((result.get("settings") or {}).get("score_threshold") or ctx.threshold)
    completion = (result.get("generation") or {}).get("completion")

    pills = [
        c.status_pill("REFUSED" if result.get("refused") else "ANSWERED",
                      "caution" if result.get("refused") else "verified"),
        c.status_pill(f"gate: {refusal.get('gate') or '—'}", "neutral"),
        c.status_pill("model not called" if completion is None else "model called",
                      "verified" if completion is None else "neutral"),
    ]
    c.write(f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin:10px 0">{"".join(pills)}</div>')

    if safety.get("signals"):
        chips = "".join(c.status_pill(s, "caution") for s in safety["signals"])
        c.write(f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">{chips}</div>')

    if hits:
        scores = [h.get("similarity_score", 0.0) for h in hits]
        labels = [str(h.get("document_id")) for h in hits]
        c.write(f'<div class="panel">{c.caliper(scores, threshold, "at-rest", labels=labels)}</div>')


def render(ctx: Context) -> None:
    c.write(c.empty_state(
        "The system does not answer every question it is asked",
        "<p>Two gates can stop a question before it reaches a model, and the model itself can "
        "decline after reading the evidence. Every rule below is <b>executed live</b> against the "
        "API — the results are real, not illustrations.</p>"
        "<p>Both refusal gates run before any model call, so this page works with no LLM key.</p>",
        glyph="safety"))

    runs = st.session_state.setdefault(STATE, {})

    for i, rule in enumerate(RULES):
        c.write('<hr class="hair">')
        left, right = st.columns([3, 2], gap="large")

        with left:
            badge = c.status_pill(f"rule {rule['id']}", "caution") if rule["expect"] == "safety" \
                else c.status_pill("evidence floor", "caution")
            c.write(
                f'<div style="display:flex;gap:8px;align-items:baseline;flex-wrap:wrap">'
                f'{badge}<span style="font-weight:600;font-size:.9375rem">{c.esc(rule["rule"])}</span></div>'
                f'<div style="font-size:.875rem;line-height:1.6;color:var(--muted);margin-top:8px;'
                f'max-width:68ch">{c.esc(rule["why"])}</div>'
                f'<div class="serif" style="font-size:.95rem;margin-top:12px;border-left:2px solid '
                f'var(--line);padding-left:12px">{c.esc(rule["question"])}</div>')

        with right:
            if st.button("Run this rule", key=f"safety-run-{i}", width="stretch"):
                with st.spinner(""):
                    _run(i, rule["question"])
                st.rerun()
            if i in runs:
                _outcome(runs[i], ctx)
            else:
                c.write('<div class="tiny" style="margin-top:8px">Not run yet.</div>')

    c.write('<hr class="hair">')
    c.write(c.error_state(
        "What this is not",
        "<p>The safety gate is a conservative pattern matcher, not a classifier: it catches the "
        "unambiguous cases and will miss paraphrases. The backstop is a system-prompt rule "
        "instructing the model to refuse the same class of request. Both layers are needed and "
        "neither is sufficient alone.</p>", glyph="alert"))
