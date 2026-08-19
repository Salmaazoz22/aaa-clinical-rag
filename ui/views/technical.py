# -*- coding: utf-8 -*-
"""Technical details, for judges and developers.

Concise and evidence-based: each component says what it is, and why it is there
rather than the obvious alternative. Live configuration values come from the
API; the design rationale is prose.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as ui
from ui.api_client import ApiError

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Embedding model", [
        ("What", "<code>abhinand/MedEmbed-base-v0.1</code>, pinned to revision "
                 "<code>7a90c502…</code>, 768 dimensions, L2-normalised."),
        ("Why", "A general-purpose encoder was measured against it on the same frozen questions "
                "and lost: on the original 10, the MiniLM baseline scored MRR 0.5625 against "
                "MedEmbed's 0.6194 with the same chunker. Clinical guideline text is dense with "
                "terminology a general encoder does not separate well."),
        ("Why pinned", "A revision hash, not a tag. Without it the same model name can silently "
                       "resolve to different weights, and every published metric becomes "
                       "unreproducible."),
    ]),
    ("Vector database", [
        ("What", "Qdrant, cosine distance, exhaustive search, one collection of 991 points."),
        ("Why", "The index was <i>migrated</i>, not rebuilt: the same float32 vectors the frozen "
                "evaluation was run against were copied in, with the chunk-ID ↔ vector binding "
                "verified before anything was written."),
        ("Why exhaustive", "991 vectors is far below Qdrant's indexing threshold, so search is "
                           "already exhaustive. Forcing it keeps that true if the corpus grows, "
                           "because the validated retriever is exhaustive cosine."),
    ]),
    ("Chunking", [
        ("What", "Atomic, page-safe chunking. 1,760 chunks produced, 991 indexed — references, "
                 "contents pages, boilerplate and title-only slides are labelled and excluded, "
                 "never deleted."),
        ("Why", "It is the single change in this project's history that raised P@1 on a "
                "pre-registered set: final20 P@1 0.400 → 0.550, MRR 0.531 → 0.664 against the "
                "previous page-buffer chunker."),
        ("Why page-safe", "A purer variant scores marginally higher on raw metrics but spans up "
                          "to 48 pages per chunk, which makes a page citation meaningless. The "
                          "page-safe variant was shipped because a citation a reader cannot turn "
                          "to is not a citation."),
    ]),
    ("Retrieval", [
        ("What", "Dense cosine top-k. No reranking, no query rewriting, no intent detection, no "
                 "keyword bonus, no filtering, no per-question logic."),
        ("Why", "Each of those was tried and measured. BM25 alone reached P@1 0.10. No hybrid "
                "weighting beat dense retrieval on a primary metric. Full cross-encoder reranking "
                "displaced correct top-1 answers on both question sets. They were rejected on "
                "evidence and the artifacts are all preserved."),
        ("Why it matters", "The retriever cannot see which question it is answering. That is what "
                           "makes the frozen evaluation meaningful rather than a measurement of "
                           "hand-tuning."),
    ]),
    ("Generation", [
        ("What", "One thin provider interface over two OpenAI-compatible endpoints. Temperature 0. "
                 "Output is a structured JSON object, not prose."),
        ("Why structured", "A free-text answer cannot be checked. The schema binds every claim to "
                           "a chunk ID and every citation to a document, section, page and score, "
                           "which is what makes the validator possible at all."),
        ("Why a failure raises", "A provider that returns <code>None</code> on failure turns into "
                                 "an empty answer with no explanation. Here a failed call raises "
                                 "and the API reports it — no answer is ever synthesised."),
    ]),
    ("Safety", [
        ("What", "A pre-retrieval pattern gate over five signal families, combined by four "
                 "blocking rules, plus a system-prompt rule instructing the model to refuse the "
                 "same class of request."),
        ("Why before the model", "So a question containing patient details is refused without "
                                 "those details being sent to a third-party API."),
        ("Why signals, not regexes", "Most single signals have legitimate general uses. "
                                     "\"What statin dose do the guidelines recommend?\" is a "
                                     "dosing question that must be answered. Blocking is a rule "
                                     "over <i>combinations</i>."),
    ]),
    ("Evidence threshold", [
        ("What", "A cosine similarity floor, default 0.75. Chunks below it are not sent to the "
                 "model; if nothing clears it, the system refuses."),
        ("Why", "The reference behaviour — pass whatever the vector store returned into the prompt "
                "— means an out-of-scope question still gets ten chunks and gets answered from "
                "them. The floor turns that into a refusal."),
        ("Honest caveat", "0.75 is a <b>starting value</b>, chosen as a conservative default. It "
                          "is not derived from a calibration study, and it is expected to be "
                          "tuned once the generation evaluation has run more times."),
    ]),
    ("Citation validation", [
        ("What", "Every answer is checked against exactly the chunks that were sent to the model: "
                 "citation resolution, document/section/page agreement, retrieval-score fidelity, "
                 "and verbatim excerpt matching."),
        ("Why 'exactly what was sent'", "A citation to a chunk that was retrieved but filtered out "
                                        "below the floor is still a citation to something the "
                                        "model never saw, so it counts as fabricated."),
        ("Why it reports rather than repairs", "A silently corrected citation is indistinguishable "
                                               "from a correct one. Findings are surfaced with "
                                               "expected and actual values so a reader can judge."),
    ]),
    ("Evaluation methodology", [
        ("What", "Three question sets, reported separately and never pooled: original 10, "
                 "held-out 18, and a pre-registered final 20."),
        ("Why separate", "Pooling them would let a gain on the set that drove the tuning decisions "
                         "hide behind a set that did not. The held-out set was written later and "
                         "kept out of every tuning decision."),
        ("Why frozen", "Each question set's SHA-256 is stamped at freeze time and re-checked "
                       "against the file and against the digest the evaluation recorded using. "
                       "Passages were validated against the source page text, not against "
                       "retrieval output."),
        ("Stated limits", "Small sets — 10 + 18 + 20 questions. No statistical significance is "
                          "claimed anywhere in this project."),
    ]),
]


def render(health: dict[str, Any] | None) -> None:
    st.markdown("# Technical details")
    st.markdown(
        '<p class="footnote" style="font-size:0.95rem;max-width:72ch">Each component, what it is, '
        "and why it is there rather than the obvious alternative. Where a choice was made on "
        "measurement, the measurement is quoted.</p>",
        unsafe_allow_html=True,
    )

    meta = None
    try:
        meta = api_client.meta()
    except ApiError as error:
        ui.backend_unavailable(error)

    if meta:
        st.markdown("## Live configuration")
        st.markdown(
            '<div class="footnote" style="margin-bottom:0.7rem">Read from the running service, '
            "not from this page.</div>",
            unsafe_allow_html=True,
        )
        gen = meta.get("generation") or {}
        ui.stat_row([
            ui.stat("Embedding", str(meta.get("model", "")).split("/")[-1],
                    f"{meta.get('dimensions')}-d · {meta.get('distance')}", "accent"),
            ui.stat("Vector store", meta.get("vector_store"),
                    (meta.get("connection") or {}).get("mode", ""), "accent"),
            ui.stat("Indexed", f"{meta.get('chunk_count'):,}" if meta.get("chunk_count") else "—",
                    meta.get("collection", ""), "green"),
            ui.stat("Retrieval top-k", gen.get("top_k", "—"), "chunks requested"),
            ui.stat("Evidence floor", gen.get("score_threshold", "—"), "min similarity", "amber"),
            ui.stat("LLM", str(gen.get("model") or "—").split("/")[-1],
                    "key configured" if gen.get("api_key_supplied") else "no key configured",
                    "green" if gen.get("api_key_supplied") else "amber"),
        ])
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

    for title, entries in SECTIONS:
        st.markdown(f"### {title}")
        body = "".join(
            f'<div class="chain-row"><div class="chain-k">{ui.esc(label)}</div>'
            f'<div class="chain-v" style="font-family:inherit;font-size:0.87rem;'
            f'line-height:1.65;word-break:normal">{text}</div></div>'
            for label, text in entries
        )
        st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("## Repository layout")
    st.markdown(
        ui.html_table(
            ["Path", "What lives there"],
            [
                ["ingestion/", "PDF extraction, cleaning, chunking"],
                ["retrieval/", "the local index and its chunk↔vector binding check"],
                ["vectordb/", "Qdrant schema, migration, retriever"],
                ["generation/", "prompts, safety, threshold, provider, parser, validator, pipeline"],
                ["api/", "the FastAPI transport layer — no core logic"],
                ["ui/", "this Streamlit client — HTTP only, no pipeline imports"],
                ["eval/", "frozen question sets, every experiment run, integrity checks"],
                ["tests/", "the test suite"],
            ],
        ),
        unsafe_allow_html=True,
    )

    if meta:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        ui.provenance_panel(meta)
