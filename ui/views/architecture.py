# -*- coding: utf-8 -*-
"""Architecture — the diagram is the page.

Hand-authored inline SVG in the token palette, drawn from the implementation as
it is: the stage order is the order in `generation/pipeline.py`, and the two
refusal exits are the two gates that exist in the code. Model, collection, chunk
count and evidence floor are read live from `/v1/meta`, so the diagram cannot
drift from the running service.
"""
from __future__ import annotations

from typing import Any

from ui import components as c
from ui.shell import Context

W, BOX_W, BOX_H, GAP = 940, 260, 54, 30
X = (W - BOX_W) / 2

# Layer colours name a role, not a decoration.
TRANSPORT, RETRIEVE, GATE = "var(--neutral)", "var(--verified)", "var(--contrast)"


def _box(y: float, title: str, sub: str, accent: str) -> str:
    return (
        f'<g><rect x="{X}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="4" '
        f'fill="var(--surface)" stroke="var(--line)"/>'
        f'<rect x="{X}" y="{y}" width="3" height="{BOX_H}" rx="1.5" fill="{accent}"/>'
        f'<text x="{X + 16}" y="{y + 22}" font-family="var(--sans)" font-size="13.5" '
        f'font-weight="600" fill="var(--ink)">{c.esc(title)}</text>'
        f'<text x="{X + 16}" y="{y + 40}" font-family="var(--mono)" font-size="10.5" '
        f'fill="var(--muted)">{c.esc(sub)}</text></g>'
    )


def _arrow(y: float, label: str) -> str:
    x = W / 2
    text = (f'<text x="{x + 12}" y="{y + GAP / 2 + 4}" font-family="var(--mono)" '
            f'font-size="9.5" fill="var(--muted)">{c.esc(label)}</text>') if label else ""
    return (f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + GAP - 8}" stroke="var(--line)" '
            f'stroke-width="1.5" marker-end="url(#ar)"/>{text}')


def _exit(y: float, side: str, title: str, sub: str) -> str:
    w, h = 250, 48
    ex = X - w - 44 if side == "left" else X + BOX_W + 44
    x1 = X if side == "left" else X + BOX_W
    x2 = ex + w if side == "left" else ex
    ey = y + BOX_H / 2 - h / 2
    return (
        f'<line x1="{x1}" y1="{y + BOX_H / 2}" x2="{x2}" y2="{y + BOX_H / 2}" '
        f'stroke="var(--aorta)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ax)"/>'
        f'<g><rect x="{ex}" y="{ey}" width="{w}" height="{h}" rx="4" fill="var(--surface)" '
        f'stroke="var(--aorta)" stroke-opacity=".45"/>'
        f'<text x="{ex + 14}" y="{ey + 20}" font-family="var(--sans)" font-size="12.5" '
        f'font-weight="600" fill="var(--aorta)">{c.esc(title)}</text>'
        f'<text x="{ex + 14}" y="{ey + 36}" font-family="var(--mono)" font-size="9.5" '
        f'fill="var(--aorta)" opacity=".85">{c.esc(sub)}</text></g>'
    )


def _diagram(ctx: Context) -> str:
    meta = ctx.meta or {}
    gen = meta.get("generation") or {}
    model = str(meta.get("model") or "MedEmbed-base-v0.1").split("/")[-1]
    dims = meta.get("dimensions") or 768
    collection = meta.get("collection") or "aaa_clinical_v1"
    chunks = meta.get("chunk_count")
    floor = gen.get("score_threshold", ctx.threshold)
    top_k = gen.get("top_k", ctx.top_k)
    llm = f"{gen.get('provider')}/{str(gen.get('model') or '').split('/')[-1]}" if gen.get("provider") else "provider"

    stages = [
        ("Clinician", "asks a guideline question", TRANSPORT),
        ("Streamlit", "HTTP client only — no pipeline code", TRANSPORT),
        ("FastAPI · POST /v1/answer", "thin transport layer", TRANSPORT),
        ("Safety gate", "patient-specific screen, pre-retrieval", GATE),
        (f"MedEmbed · {model}", f"query → {dims}-d vector, L2-normalised", RETRIEVE),
        (f"Qdrant · {collection}", f"{chunks:,} vectors · exhaustive cosine" if chunks else "cosine", RETRIEVE),
        ("Evidence threshold", f"keep similarity ≥ {floor}, of top-{top_k}", GATE),
        (f"LLM · {llm}", "answers ONLY from the chunks it was sent", TRANSPORT),
        ("Citation validator", "every citation checked against those chunks", RETRIEVE),
        ("Grounded answer", "recommendation + citations + confidence", RETRIEVE),
    ]
    labels = ["", "HTTP", "", "not blocked", "query vector",
              f"top-{top_k} hits", "usable evidence", "structured JSON", "validated"]

    parts, ys, y = [], [], 14.0
    for i, (title, sub, accent) in enumerate(stages):
        ys.append(y)
        parts.append(_box(y, title, sub, accent))
        if i < len(stages) - 1:
            parts.append(_arrow(y + BOX_H, labels[i]))
        y += BOX_H + GAP

    parts.append(_exit(ys[3], "left", "Refused — patient-specific", "built locally · model never called"))
    parts.append(_exit(ys[6], "right", "Refused — below floor", "names the passages it rejected"))

    return (
        f'<div class="scroll-x"><svg viewBox="0 0 {W} {y + 8:.0f}" width="100%" '
        f'style="max-width:{W}px;min-width:620px" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Request path from the clinician through Streamlit, FastAPI, the safety gate, '
        f'MedEmbed, Qdrant, the evidence threshold, the language model and the citation validator '
        f'to a grounded answer, with two refusal exits.">'
        f'<defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 7 3.5, 0 7" fill="var(--line)"/></marker>'
        f'<marker id="ax" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 7 3.5, 0 7" fill="var(--aorta)"/></marker></defs>'
        f'{"".join(parts)}</svg></div>'
    )


NOTES = (
    ("The UI is a client, not a second pipeline",
     "Streamlit imports no retrieval, generation or vector-store code. It speaks HTTP to FastAPI "
     "and renders what comes back. There is exactly one retrieval implementation in this project, "
     "and a test enforces it by parsing every UI source."),
    ("The safety gate sits above retrieval, not inside it",
     "It decides whether the model is called at all. It never changes ranking, scoring, or which "
     "chunks come back — the retrieval path still cannot see which question it is answering, which "
     "is what makes the frozen evaluation meaningful."),
    ("The evidence threshold is where most systems just answer",
     "A conventional RAG pipeline passes whatever the vector store returned straight into the "
     "prompt. Here every hit must clear a similarity floor to become evidence, and a question "
     "where nothing clears it is refused rather than answered from adjacent text."),
    ("Validation runs against exactly what was sent",
     "A citation to a chunk that was retrieved but filtered out below the floor is still a citation "
     "to something the model never saw, so it counts as fabricated and is reported as an error."),
)


def render(ctx: Context) -> None:
    c.write(f'<div class="panel">{_diagram(ctx)}</div>')

    legend = "".join([
        c.status_pill("transport & orchestration", "neutral"),
        c.status_pill("retrieval & verification", "verified"),
        c.status_pill("gate — can refuse here", "caution"),
    ])
    c.write(f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">{legend}</div>')

    c.write('<hr class="hair"><div class="eyebrow">Why it is shaped this way</div>'
            '<div style="height:12px"></div>')
    import streamlit as st
    cols = st.columns(2, gap="large")
    for i, (title, body) in enumerate(NOTES):
        with cols[i % 2]:
            c.write(f'<div class="panel"><div style="font-weight:600;font-size:.9375rem">'
                    f'{c.esc(title)}</div><div style="font-size:.875rem;line-height:1.65;'
                    f'color:var(--muted);margin-top:8px">{c.esc(body)}</div></div>')

    c.write('<hr class="hair"><div class="eyebrow">API surface</div><div style="height:12px"></div>')
    c.write(c.data_table(
        ["Endpoint", "Returns", "Used here for"],
        [["GET /health", "API, Qdrant, index, LLM-key status", "the rail telemetry"],
         ["GET /v1/meta", "model, revision, collection, digests", "provenance, this diagram"],
         ["POST /v1/answer", "the full pipeline, one call", "every answer and refusal"],
         ["GET /v1/chunks/{id}", "one chunk's stored payload", "full passage text"],
         ["GET /v1/corpus", "the guideline documents", "the Sources page"],
         ["GET /v1/evaluation", "the frozen evaluation", "the Evaluation page"]]))
