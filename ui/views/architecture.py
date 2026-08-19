# -*- coding: utf-8 -*-
"""The architecture page.

The diagram is hand-authored SVG and reflects the implementation as it is: the
stage order is the order in `generation/pipeline.py`, the two refusal exits are
the two gates that actually exist, and the labels carry the real model, the real
collection and the real threshold, read from `GET /v1/meta` where available.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as ui
from ui.api_client import ApiError
from ui.theme import ACCENT, AMBER, BORDER, BORDER_STRONG, GREEN, INK, INK_FAINT, INK_MUTED, MONO, SANS, SERIF

W = 940
BOX_W = 250
BOX_H = 52
X = (W - BOX_W) / 2
GAP = 30


def _box(y: float, title: str, subtitle: str, *, fill: str = "#FFFFFF",
         stroke: str = BORDER_STRONG, accent: str | None = None) -> str:
    left = f'<rect x="{X}" y="{y}" width="4" height="{BOX_H}" fill="{accent}" rx="2"/>' if accent else ""
    return f"""
  <g>
    <rect x="{X}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="5"
          fill="{fill}" stroke="{stroke}" stroke-width="1"/>
    {left}
    <text x="{X + 16}" y="{y + 22}" font-family="{SANS}" font-size="13.5"
          font-weight="600" fill="{INK}">{ui.esc(title)}</text>
    <text x="{X + 16}" y="{y + 39}" font-family="{MONO}" font-size="10.5"
          fill="{INK_MUTED}">{ui.esc(subtitle)}</text>
  </g>"""


def _arrow(y: float, label: str = "") -> str:
    x = W / 2
    text = (
        f'<text x="{x + 12}" y="{y + GAP / 2 + 4}" font-family="{MONO}" font-size="10"'
        f' fill="{INK_FAINT}">{ui.esc(label)}</text>'
        if label
        else ""
    )
    return (
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + GAP - 8}" stroke="{BORDER_STRONG}"'
        f' stroke-width="1.5" marker-end="url(#arrowhead)"/>{text}'
    )


def _exit(y: float, side: str, title: str, subtitle: str) -> str:
    """A refusal branch leaving the spine sideways."""
    exit_w, exit_h = 250, 50
    if side == "left":
        exit_x = X - exit_w - 46
        line = (
            f'<line x1="{X}" y1="{y + BOX_H / 2}" x2="{exit_x + exit_w}" y2="{y + BOX_H / 2}"'
            f' stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowamber)"/>'
        )
    else:
        exit_x = X + BOX_W + 46
        line = (
            f'<line x1="{X + BOX_W}" y1="{y + BOX_H / 2}" x2="{exit_x}" y2="{y + BOX_H / 2}"'
            f' stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowamber)"/>'
        )
    ey = y + BOX_H / 2 - exit_h / 2
    return f"""{line}
  <g>
    <rect x="{exit_x}" y="{ey}" width="{exit_w}" height="{exit_h}" rx="5"
          fill="#FDF3E2" stroke="#EFD9AE" stroke-width="1"/>
    <text x="{exit_x + 14}" y="{ey + 21}" font-family="{SANS}" font-size="12.5"
          font-weight="600" fill="{AMBER}">{ui.esc(title)}</text>
    <text x="{exit_x + 14}" y="{ey + 37}" font-family="{MONO}" font-size="10"
          fill="{AMBER}">{ui.esc(subtitle)}</text>
  </g>"""


def _diagram(meta: dict[str, Any] | None) -> str:
    model = "MedEmbed-base-v0.1"
    dims = 768
    collection = "aaa_clinical_v1"
    chunks = None
    floor = 0.75
    top_k = 5
    llm = "configured provider"

    if meta:
        model = str(meta.get("model") or model).split("/")[-1]
        dims = meta.get("dimensions") or dims
        collection = meta.get("collection") or collection
        chunks = meta.get("chunk_count")
        gen = meta.get("generation") or {}
        floor = gen.get("score_threshold", floor)
        top_k = gen.get("top_k", top_k)
        if gen.get("provider"):
            llm = f"{gen['provider']}/{str(gen.get('model') or '').split('/')[-1]}"

    chunk_txt = f"{chunks:,} vectors · cosine" if chunks else f"{dims}-d · cosine"

    stages = [
        ("Clinician", "asks a guideline question", ACCENT),
        ("Streamlit UI", "HTTP client only — no pipeline code", ACCENT),
        ("FastAPI  ·  POST /v1/answer", "thin transport layer", ACCENT),
        ("Safety gate", "patient-specific screen, pre-retrieval", AMBER),
        (f"MedEmbed  ·  {model}", f"query → {dims}-d vector, L2-normalised", GREEN),
        (f"Qdrant  ·  {collection}", chunk_txt, GREEN),
        ("Evidence threshold", f"keep similarity ≥ {floor}, of top-{top_k}", AMBER),
        (f"LLM  ·  {llm}", "answers ONLY from the chunks it was sent", ACCENT),
        ("Citation validator", "every citation checked against those chunks", GREEN),
        ("Grounded answer", "recommendation + citations + confidence", GREEN),
    ]

    arrow_labels = [
        "", "HTTP", "", "not blocked", "query vector",
        "top-k hits", "usable evidence", "structured JSON", "validated",
    ]

    parts: list[str] = []
    y = 14.0
    positions: list[float] = []
    for index, (title, subtitle, accent) in enumerate(stages):
        positions.append(y)
        parts.append(_box(y, title, subtitle, accent=accent))
        if index < len(stages) - 1:
            parts.append(_arrow(y + BOX_H, arrow_labels[index]))
        y += BOX_H + GAP

    # Refusal branches leave the two gates that actually exist.
    parts.append(_exit(positions[3], "left", "Refused — patient-specific",
                       "built locally · model never called"))
    parts.append(_exit(positions[6], "right", "Refused — below evidence floor",
                       "names the passages it rejected"))

    height = y + 10
    return f"""
<div style="overflow-x:auto">
<svg viewBox="0 0 {W} {height:.0f}" width="100%" style="max-width:{W}px;min-width:640px"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Request path from the clinician through Streamlit, FastAPI, the safety gate,
                 MedEmbed, Qdrant, the evidence threshold, the language model and the citation
                 validator to a grounded answer, with two refusal exits.">
  <defs>
    <marker id="arrowhead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <polygon points="0 0, 7 3.5, 0 7" fill="{BORDER_STRONG}"/>
    </marker>
    <marker id="arrowamber" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <polygon points="0 0, 7 3.5, 0 7" fill="{AMBER}"/>
    </marker>
  </defs>
  {"".join(parts)}
</svg>
</div>"""


NOTES = [
    ("The UI is a client, not a second pipeline",
     "Streamlit imports no retrieval, generation or vector-store code. It speaks HTTP to FastAPI "
     "and renders what comes back. There is exactly one retrieval implementation in this project."),
    ("The safety gate sits above retrieval, not inside it",
     "It decides whether the model is called at all. It never changes ranking, scoring, or which "
     "chunks come back — the retrieval path still cannot see which question it is answering, "
     "which is what makes the frozen evaluation meaningful."),
    ("The evidence threshold is where most systems just answer",
     "A conventional RAG pipeline passes whatever the vector store returned straight into the "
     "prompt. Here every hit must clear a similarity floor to become evidence, and a question "
     "where nothing clears it is refused rather than answered from adjacent text."),
    ("The model is given evidence, and only evidence",
     "Chunk text is sent in full — never truncated — because the model is asked to quote verbatim "
     "excerpts that are then checked against the chunk they are attributed to. Truncating the "
     "evidence would make those excerpts unverifiable."),
    ("Validation runs against exactly what was sent",
     "A citation to a chunk that was retrieved but filtered out below the floor is still a "
     "citation to something the model never saw, so it counts as fabricated and is reported as "
     "an error."),
    ("Nothing is silently repaired",
     "The validator reports; it does not rewrite. The one exception is the fixed safety "
     "disclaimer, which is normalised — and the substitution is recorded as a finding, so it is "
     "visible rather than silent."),
]


def render(health: dict[str, Any] | None) -> None:
    st.markdown("# Architecture")
    st.markdown(
        '<p class="footnote" style="font-size:0.95rem;max-width:70ch">The request path, as '
        "implemented. Stage order is the order in <code>generation/pipeline.py</code>; the two "
        "refusal exits are the two gates that exist in the code. Model, collection, chunk count "
        "and evidence floor are read live from the API.</p>",
        unsafe_allow_html=True,
    )

    meta = None
    try:
        meta = api_client.meta()
    except ApiError as error:
        ui.backend_unavailable(error)
        st.markdown(
            '<div class="footnote">The diagram below is still shown, but with the configured '
            "defaults rather than live values.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="card">{_diagram(meta)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="badge-row" style="margin-bottom:1.5rem">'
        f'<span class="badge accent">■ transport &amp; orchestration</span>'
        f'<span class="badge ok">■ retrieval &amp; verification</span>'
        f'<span class="badge warn">■ gate — can refuse here</span>'
        f'<span class="badge warn">┄ refusal exit</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## Why it is shaped this way")

    cols = st.columns(2, gap="large")
    for index, (title, body) in enumerate(NOTES):
        with cols[index % 2]:
            st.markdown(
                f'<div class="card"><div class="card-label">{ui.esc(title)}</div>'
                f'<div class="footnote" style="font-size:0.85rem;line-height:1.65">'
                f"{ui.esc(body)}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## The API surface")
    st.markdown(
        ui.html_table(
            ["Endpoint", "Purpose", "Used by this UI for"],
            [
                ["GET /health", "API, Qdrant, index and LLM-key status", "the sidebar status strip"],
                ["GET /v1/meta", "model, revision, collection, index digests", "badges, provenance, this diagram"],
                ["POST /v1/answer", "the full pipeline, one call", "every answer and refusal"],
                ["GET /v1/chunks/{id}", "one chunk's stored payload", "full evidence text on demand"],
                ["GET /v1/corpus", "the guideline documents behind the index", "the Guidelines page"],
                ["GET /v1/evaluation", "the frozen evaluation artifacts", "the Evaluation page"],
            ],
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="footnote" style="margin-top:0.7rem">Interactive OpenAPI documentation is '
        f'served by the backend at <code>{ui.esc(api_client.base_url())}/docs</code>.</div>',
        unsafe_allow_html=True,
    )
