# -*- coding: utf-8 -*-
"""Evaluation — the frozen retrieval results.

Every number is served by `GET /v1/evaluation`, which reads the published
artifacts off disk and returns them verbatim. Nothing is recomputed, nothing is
averaged across datasets, and nothing is rounded differently from the way it was
published. The page also shows the SHA-256 of the bytes the API read.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as c
from ui.api_client import ApiError
from ui.shell import Context

SHIPPED = "V1_atomic_pagesafe"
METRICS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10")

SETS = (
    ("original10", "Original 10", "Written before any experiment ran."),
    ("heldout18", "Held-out 18", "Written later and kept out of every tuning decision."),
    ("final20", "Final 20",
     "The pre-registered set the shipped configuration is judged on. Passages were validated "
     "against the source page text, not against retrieval output."),
)

LABELS = {
    "baseline_production": "Baseline · page-buffer chunker",
    SHIPPED: "V1 atomic · shipped",
    "V2_atomic_pure": "V2 atomic · not shipped",
}

#: One line of plain language per metric. Interpretation, not data.
READS = {
    "P@1": "share of questions whose rank-1 chunk is a gold passage",
    "MRR": "mean reciprocal rank of the first relevant passage",
    "Recall@5": "share of gold passages found in the top 5",
    "Recall@10": "share of gold passages found in the top 10",
}


def _m(metrics: dict, dataset: str, config: str) -> dict:
    return ((metrics.get(dataset) or {}).get(config) or {}).get("metrics") or {}


def _fmt(value: Any) -> str:
    return c.num(value, 4)


def render(ctx: Context) -> None:
    try:
        data = api_client.evaluation()
    except ApiError as error:
        c.write(c.error_state("Evaluation unavailable", f"<p>{c.esc(error.message)}</p>"))
        return

    c.write(c.empty_state(
        "Frozen results",
        "<p>Everything below is a frozen artifact, served verbatim from the files the published "
        "evaluation wrote. It is <b>not</b> a live re-run and it does not reflect questions asked "
        "on the Ask page — those are live and are not scored against any gold standard.</p>",
        glyph="info"))

    metrics = data.get("metrics") or {}
    shipped = _m(metrics, "final20", SHIPPED)
    base = _m(metrics, "final20", "baseline_production")

    if shipped:
        c.write('<hr class="hair"><div class="eyebrow">Shipped configuration · final20</div>')
        tiles = []
        for key, tone in (("P@1", "accent"), ("MRR", "accent"), ("Recall@5", ""), ("Recall@10", "verified")):
            delta = ""
            try:
                delta = f"{float(shipped[key]) - float(base[key]):+.4f} vs baseline"
            except (KeyError, TypeError, ValueError):
                delta = READS.get(key, "")
            tiles.append(c.metric_tile(key, _fmt(shipped.get(key)), delta, tone))
        tiles.append(c.metric_tile(
            "Relevant @1", f"{shipped.get('Relevant_Top1')}/{shipped.get('n_queries')}",
            READS["P@1"], "verified"))
        c.tile_row(tiles)

    # -- per-set tables ---------------------------------------------------
    for key, title, blurb in SETS:
        configs = metrics.get(key) or {}
        if not configs:
            continue
        c.write(f'<hr class="hair"><div class="eyebrow">{c.esc(title)}</div>'
                f'<div class="tiny" style="margin:4px 0 12px;max-width:68ch">{c.esc(blurb)}</div>')
        names = list(configs)
        rows, hl = [], None
        for i, name in enumerate(names):
            values = _m(metrics, key, name)
            if name == SHIPPED:
                hl = i
            rows.append([
                LABELS.get(name, name), values.get("n_queries"),
                *[_fmt(values.get(m)) for m in METRICS],
                f"{values.get('Relevant_Top1')}/{values.get('n_queries')}",
            ])
        c.write(c.data_table(["Configuration", "n", *METRICS, "Rel@1"], rows, highlight=hl))

    # -- evidence statistics ----------------------------------------------
    stats = data.get("evidence_statistics") or {}
    if stats:
        total = stats.get("questions")
        c.write('<hr class="hair"><div class="eyebrow">Where the relevant evidence landed</div>'
                '<div style="height:12px"></div>')
        missing = stats.get("no_relevant_evidence_in_top_10") or []
        c.tile_row([
            c.metric_tile("Questions", total, "in the final set"),
            c.metric_tile("Relevant @1", f"{stats.get('relevant_in_top_1')}/{total}", "", "verified"),
            c.metric_tile("Relevant @5", f"{stats.get('relevant_in_top_5')}/{total}", "", "verified"),
            c.metric_tile("Relevant @10", f"{stats.get('relevant_in_top_10')}/{total}", "", "verified"),
            c.metric_tile("No evidence", ", ".join(missing) or "none",
                          "anywhere in the top 10", "aorta"),
        ])
        c.write('<div class="tiny" style="margin-top:10px;max-width:68ch">The one question with no '
                'relevant chunk anywhere in its top 10 is also the question the evidence floor '
                'refuses — reached by a threshold, not by knowing which question it is.</div>')

    # -- interpretation / limitations -------------------------------------
    c.write('<hr class="hair">')
    left, right = st.columns(2, gap="large")
    with left:
        c.write('<div class="eyebrow">Interpretation</div>')
        text = data.get("final_interpretation")
        if text:
            c.write(f'<div class="panel"><div style="font-size:.875rem;line-height:1.65;'
                    f'color:var(--muted)">{c.esc(text)}</div></div>')
    with right:
        c.write('<div class="eyebrow">Stated limitations</div>')
        items = data.get("limitations") or []
        if items:
            lis = "".join(f'<li style="margin-bottom:8px">{c.esc(x)}</li>' for x in items)
            c.write(f'<div class="panel"><ul style="font-size:.875rem;line-height:1.6;'
                    f'color:var(--muted);padding-left:18px;margin:0">{lis}</ul></div>')

    # -- provenance --------------------------------------------------------
    c.write('<hr class="hair"><div class="eyebrow">Provenance</div><div style="height:12px"></div>')
    rows = [
        ("Embedding model", data.get("embedding_model")),
        ("Embedding revision", data.get("embedding_revision")),
        ("Retrieval", data.get("retrieval")),
        ("Evaluation depth", data.get("eval_depth")),
        ("Shipped configuration", data.get("shipped_config")),
    ]
    rows += [(f"Gold SHA-256 · {k}", v) for k, v in (data.get("gold_sha256") or {}).items()]
    rows += [(f"Artifact SHA-256 · {k}", v) for k, v in (data.get("provenance") or {}).items() if v]
    c.write(c.panel("", c.definition_list(rows, prose_keys=("Retrieval",))))

    integrity = data.get("integrity")
    if integrity:
        summary = integrity.get("summary") or {}
        passed, failed = summary.get("pass", 0), summary.get("fail", 0)
        c.write('<div class="eyebrow" style="margin-top:20px">Reproducibility checks</div>'
                '<div style="height:12px"></div>')
        c.tile_row([
            c.metric_tile("Passed", passed, "", "verified"),
            c.metric_tile("Failed", failed, "shown, not hidden", "aorta" if failed else "verified"),
        ])
        with st.expander(f"All {len(integrity.get('checks') or [])} checks", expanded=bool(failed)):
            for check in integrity.get("checks") or []:
                ok = str(check.get("status", "")).lower() == "pass"
                c.write(
                    f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;'
                    f'border-bottom:1px solid var(--line)">'
                    f'{c.status_pill("PASS" if ok else "FAIL", "verified" if ok else "caution")}'
                    f'<span style="font-size:.8125rem">{c.esc(check.get("check"))}</span></div>')
