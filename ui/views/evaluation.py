# -*- coding: utf-8 -*-
"""The frozen retrieval evaluation.

Every number on this page is served by `GET /v1/evaluation`, which reads
`eval/final_evaluation_results.json` and `eval/final_evaluation_summary.json`
off disk and returns them verbatim. Nothing is recomputed here, nothing is
averaged across datasets, and nothing is rounded differently from the way it
was published. The page also shows the SHA-256 of the bytes the API read, so a
reader can check that what is on screen is the frozen artifact.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as ui
from ui.api_client import ApiError

METRIC_KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10")

DATASETS = (
    ("original10", "Original 10", "The first question set, written before any experiment ran."),
    ("heldout18", "Held-out 18", "Written later and kept out of every tuning decision."),
    ("final20", "Final 20",
     "The pre-registered set the shipped configuration is judged on. Passages were validated "
     "against the source page text, not against retrieval output."),
)

SHIPPED = "V1_atomic_pagesafe"

CONFIG_LABEL = {
    "baseline_production": "Baseline (page-buffer chunker, historical)",
    "V1_atomic_pagesafe": "V1 atomic, page-safe  ·  SHIPPED",
    "V2_atomic_pure": "V2 atomic, pure (not shipped)",
}


def render(health: dict[str, Any] | None) -> None:
    st.markdown("# Evaluation")
    st.markdown(
        '<p class="footnote" style="font-size:0.95rem;max-width:70ch">Retrieval quality, measured '
        "on three separate question sets that are reported separately and never pooled. These are "
        "the published results — this page cannot produce any other number.</p>",
        unsafe_allow_html=True,
    )

    try:
        data = api_client.evaluation()
    except ApiError as error:
        ui.backend_unavailable(error)
        return

    ui.notice(
        "info",
        "Frozen evaluation results",
        "<p>Everything below is a <b>frozen</b> artifact, served verbatim by the API from the "
        "files the published evaluation wrote. It is <b>not</b> a live re-run and does not reflect "
        "the demo questions you may have asked on the Ask page — those are live results and are "
        "not scored against any gold standard.</p>",
    )

    metrics = data.get("metrics") or {}

    # -- headline ---------------------------------------------------------
    shipped_final = _metrics_for(metrics, "final20", SHIPPED)
    baseline_final = _metrics_for(metrics, "final20", "baseline_production")
    if shipped_final:
        st.markdown("## Shipped configuration on the pre-registered set")
        st.markdown(
            f'<div class="footnote" style="margin-bottom:0.8rem">'
            f'<code>final20</code> · {shipped_final.get("n_queries")} questions · '
            f"{ui.esc(CONFIG_LABEL.get(SHIPPED, SHIPPED))}</div>",
            unsafe_allow_html=True,
        )
        ui.stat_row([
            ui.stat("P@1", shipped_final.get("P@1"), _delta(shipped_final, baseline_final, "P@1"), "accent"),
            ui.stat("MRR", shipped_final.get("MRR"), _delta(shipped_final, baseline_final, "MRR"), "accent"),
            ui.stat("Recall@5", shipped_final.get("Recall@5"), _delta(shipped_final, baseline_final, "Recall@5")),
            ui.stat("Recall@10", shipped_final.get("Recall@10"), _delta(shipped_final, baseline_final, "Recall@10")),
            ui.stat("Relevant top-1", f"{shipped_final.get('Relevant_Top1')}/{shipped_final.get('n_queries')}",
                    "questions whose rank-1 chunk is a gold passage", "green"),
        ])
        st.markdown(
            '<div class="footnote" style="margin-top:0.6rem">Deltas are against the historical '
            "page-buffer baseline on the same question set, not against a different set.</div>",
            unsafe_allow_html=True,
        )

    # -- per-dataset tables ------------------------------------------------
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## Every configuration, every set")

    for key, title, blurb in DATASETS:
        configs = metrics.get(key) or {}
        if not configs:
            continue
        st.markdown(f"### {title}")
        st.markdown(f'<div class="footnote" style="margin-bottom:0.6rem">{ui.esc(blurb)}</div>',
                    unsafe_allow_html=True)

        names = list(configs)
        rows = []
        shipped_index = None
        for index, name in enumerate(names):
            values = _metrics_for(metrics, key, name)
            if name == SHIPPED:
                shipped_index = index
            rows.append([
                CONFIG_LABEL.get(name, name),
                values.get("n_queries"),
                *[_fmt(values.get(metric)) for metric in METRIC_KEYS],
                f"{values.get('Relevant_Top1')}/{values.get('n_queries')}",
                f"{values.get('Answering@5')}/{values.get('n_queries')}",
            ])

        st.markdown(
            ui.html_table(
                ["Configuration", "n", *METRIC_KEYS, "Rel@1", "Ans@5"],
                rows,
                shipped_row=shipped_index,
            ),
            unsafe_allow_html=True,
        )
        _bar_compare(configs, key)
        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

    # -- evidence statistics ----------------------------------------------
    stats = data.get("evidence_statistics") or {}
    if stats:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown("## Where the relevant evidence landed")
        total = stats.get("questions")
        ui.stat_row([
            ui.stat("Questions", total, "in the final set"),
            ui.stat("Relevant in top-1", f"{stats.get('relevant_in_top_1')}/{total}", "", "green"),
            ui.stat("Relevant in top-5", f"{stats.get('relevant_in_top_5')}/{total}", "", "green"),
            ui.stat("Relevant in top-10", f"{stats.get('relevant_in_top_10')}/{total}", "", "green"),
            ui.stat("No relevant evidence", ", ".join(stats.get("no_relevant_evidence_in_top_10") or []) or "none",
                    "anywhere in the top 10", "amber"),
        ])
        st.markdown(
            '<div class="footnote" style="margin-top:0.6rem">The one question with no relevant '
            "chunk anywhere in its top 10 is also the question the shipped evidence floor refuses "
            "— reached by a threshold, not by knowing which question it is.</div>",
            unsafe_allow_html=True,
        )

    # -- interpretation and limitations -----------------------------------
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### Interpretation")
        interpretation = data.get("final_interpretation")
        if interpretation:
            st.markdown(
                f'<div class="card"><div class="footnote" style="font-size:0.85rem;line-height:1.7">'
                f"{ui.esc(interpretation)}</div></div>",
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("### Stated limitations")
        limitations = data.get("limitations") or []
        if limitations:
            items = "".join(
                f'<li style="margin-bottom:0.5rem">{ui.esc(item)}</li>' for item in limitations
            )
            st.markdown(
                f'<div class="card"><ul class="footnote" style="font-size:0.85rem;line-height:1.6;'
                f'padding-left:1.1rem;margin:0">{items}</ul></div>',
                unsafe_allow_html=True,
            )

    # -- provenance --------------------------------------------------------
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## Evaluation provenance")

    provenance = data.get("provenance") or {}
    gold = data.get("gold_sha256") or {}
    rows: list[tuple[str, str]] = [
        ("Embedding model", ui.esc(data.get("embedding_model"))),
        ("Embedding revision", ui.esc(data.get("embedding_revision"))),
        ("Retrieval", ui.esc(data.get("retrieval"))),
        ("Evaluation depth", ui.esc(data.get("eval_depth"))),
        ("Shipped configuration", ui.esc(data.get("shipped_config"))),
    ]
    rows += [(f"Gold SHA-256 · {name}", ui.esc(value)) for name, value in gold.items()]
    rows += [(f"Artifact SHA-256 · {name}", ui.esc(value)) for name, value in provenance.items() if value]

    body = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in rows)
    st.markdown(
        f'<div class="card"><div class="card-label">What produced these numbers</div>'
        f'<div class="kv">{body}</div>'
        f'<div class="footnote" style="margin-top:0.9rem">The artifact digests are of the exact '
        f"bytes the API read to build this page. The gold digests are the ones stamped when each "
        f"question set was frozen.</div></div>",
        unsafe_allow_html=True,
    )

    rule = data.get("heldout_selection_rule")
    if rule:
        st.markdown(
            f'<div class="card tight"><div class="card-label">Held-out selection rule</div>'
            f'<div class="footnote">{ui.esc(rule)}</div></div>',
            unsafe_allow_html=True,
        )

    _integrity(data.get("integrity"))


def _integrity(integrity: dict[str, Any] | None) -> None:
    if not integrity:
        return
    summary = integrity.get("summary") or {}
    passed, failed = summary.get("pass", 0), summary.get("fail", 0)
    st.markdown("## Reproducibility checks")
    st.markdown(
        f'<div class="footnote" style="margin-bottom:0.7rem">Reported by the project\'s own '
        f"integrity script and served through the API. Failures are shown, not hidden.</div>",
        unsafe_allow_html=True,
    )
    ui.stat_row([
        ui.stat("Checks passed", passed, "", "green"),
        ui.stat("Checks failed", failed, "", "red" if failed else "green"),
    ])

    checks = integrity.get("checks") or []
    rows = []
    for check in checks:
        state = str(check.get("status", "")).lower()
        rows.append([("PASS" if state == "pass" else "FAIL"), check.get("check")])
    if rows:
        with st.expander(f"All {len(rows)} integrity checks", expanded=bool(failed)):
            for state, name in rows:
                cls = "y" if state == "PASS" else "n"
                st.markdown(
                    f'<div style="padding:0.3rem 0;border-bottom:1px solid #E1E5EC">'
                    f'<span class="check {cls}">{state}</span> '
                    f'<span style="font-size:0.85rem">{ui.esc(name)}</span></div>',
                    unsafe_allow_html=True,
                )


def _metrics_for(metrics: dict[str, Any], dataset: str, config: str) -> dict[str, Any]:
    entry = (metrics.get(dataset) or {}).get(config) or {}
    return entry.get("metrics") or {}


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> str:
    try:
        difference = float(current[key]) - float(baseline[key])
    except (KeyError, TypeError, ValueError):
        return ""
    return f"{difference:+.4f} vs baseline"


def _bar_compare(configs: dict[str, Any], dataset_key: str) -> None:
    """Compact P@1 / MRR comparison. Plain HTML bars — no chart library, no
    auto-scaling surprises, and the 0–1 axis is the real one for both metrics."""
    rows = []
    for name, entry in configs.items():
        values = entry.get("metrics") or {}
        for metric, tone in (("P@1", "used"), ("MRR", "dropped")):
            value = values.get(metric)
            try:
                width = max(0.0, min(1.0, float(value))) * 100
            except (TypeError, ValueError):
                continue
            label = f"{metric}" if name != SHIPPED else f"{metric} ★"
            rows.append(
                f'<div class="bar-row">'
                f'<div class="bar-lab">{ui.esc(label)}</div>'
                f'<div class="bar-track"><div class="bar-fill {tone}" '
                f'style="width:{width:.1f}%"></div></div>'
                f'<div class="bar-val {tone}">{_fmt(value)}</div></div>'
            )
        rows.append(
            f'<div class="footnote" style="margin:-0.1rem 0 0.6rem 0">'
            f"{ui.esc(CONFIG_LABEL.get(name, name))}</div>"
        )
    if rows:
        with st.expander(f"P@1 / MRR bars — {dataset_key}", expanded=False):
            st.markdown(f'<div class="bars">{"".join(rows)}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="footnote">Green = P@1, grey = MRR. Both are 0–1, so the bars are '
                "directly comparable. ★ marks the shipped configuration.</div>",
                unsafe_allow_html=True,
            )
