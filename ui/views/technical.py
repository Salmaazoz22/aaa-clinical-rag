# -*- coding: utf-8 -*-
"""Technical details — a definition-list layout for judges and developers.

Each component says what it is and why it is there rather than the obvious
alternative. Where a choice was made on measurement, the measurement is quoted.
Live configuration is read from `/v1/meta`; raw payloads go behind an expander,
never in the primary view.
"""
from __future__ import annotations

import streamlit as st

from ui import components as c
from ui.shell import Context

SECTIONS = (
    ("Ingestion", (
        ("Corpus", "Four authoritative AAA guidelines: ESVS 2024, NICE NG156, SVS 2018, USPSTF 2019."),
        ("Chunking", "Atomic, page-safe. 1,760 chunks produced, 991 indexed — references, contents "
                     "pages, boilerplate and title-only slides are labelled and excluded, never deleted."),
        ("Why this chunker", "It is the single change in this project's history that raised P@1 on a "
                             "pre-registered set: final20 P@1 0.400 → 0.550, MRR 0.531 → 0.664."),
        ("Why page-safe", "A purer variant scores marginally higher but spans up to 48 pages per "
                          "chunk, which makes a page citation meaningless. A citation a reader "
                          "cannot turn to is not a citation."),
    )),
    ("Embedding", (
        ("Model", "abhinand/MedEmbed-base-v0.1, pinned to revision 7a90c502, 768-d, L2-normalised."),
        ("Why", "A general-purpose encoder was measured against it on the same frozen questions and "
                "lost: MiniLM scored MRR 0.5625 against MedEmbed's 0.6194 with the same chunker."),
        ("Why a revision hash", "Not a tag. Without it the same model name can silently resolve to "
                                "different weights, and every published metric becomes unreproducible."),
    )),
    ("Retrieval", (
        ("Method", "Dense cosine top-k over Qdrant. Exhaustive search."),
        ("Not used", "No reranking, no query rewriting, no intent detection, no keyword bonus, no "
                     "filtering, no per-question logic."),
        ("Why not", "Each was tried and measured. BM25 alone reached P@1 0.10. No hybrid weighting "
                    "beat dense retrieval on a primary metric. Full cross-encoder reranking "
                    "displaced correct top-1 answers on both question sets. All artifacts preserved."),
        ("Why it matters", "The retriever cannot see which question it is answering. That is what "
                           "makes the frozen evaluation a measurement rather than a record of "
                           "hand-tuning."),
    )),
    ("Generation", (
        ("Interface", "One thin provider interface over two OpenAI-compatible endpoints. "
                      "Temperature 0."),
        ("Output", "A structured JSON object, not prose."),
        ("Why structured", "Free text cannot be checked. The schema binds every claim to a chunk ID "
                           "and every citation to a document, section, page and score — which is "
                           "what makes the validator possible at all."),
        ("Why a failure raises", "A provider that returns None on failure becomes an empty answer "
                                 "with no explanation. Here a failed call raises and the API reports "
                                 "it; no answer is ever synthesised."),
    )),
    ("Safety", (
        ("Gate", "A pre-retrieval pattern gate over five signal families, combined by four blocking "
                 "rules, plus a system-prompt rule instructing the model to refuse the same class."),
        ("Why before the model", "So a question containing patient details is refused without those "
                                 "details being sent to a third-party API."),
        ("Why signals, not regexes", "Most single signals have legitimate general uses. \"What "
                                     "statin dose do the guidelines recommend?\" must be answered. "
                                     "Blocking is a rule over combinations."),
        ("Evidence floor", "A cosine similarity floor. Chunks below it are not sent to the model; if "
                           "nothing clears it, the system refuses."),
        ("Honest caveat", "The floor is a starting value chosen as a conservative default. It is not "
                          "derived from a calibration study and is expected to be tuned once the "
                          "generation evaluation has run more times."),
    )),
    ("Validation", (
        ("Checks", "Citation resolution, document/section/page agreement, retrieval-score fidelity, "
                   "and verbatim excerpt matching — against exactly the chunks that were sent."),
        ("Why 'exactly what was sent'", "A citation to a chunk that was retrieved but filtered out "
                                        "below the floor is still a citation to something the model "
                                        "never saw, so it counts as fabricated."),
        ("Reports, never repairs", "A silently corrected citation is indistinguishable from a "
                                   "correct one. Findings are surfaced with expected and actual "
                                   "values so a reader can judge."),
    )),
    ("Evaluation", (
        ("Design", "Three question sets, reported separately and never pooled: original 10, "
                   "held-out 18, pre-registered final 20."),
        ("Why separate", "Pooling would let a gain on the set that drove tuning decisions hide "
                         "behind a set that did not."),
        ("Why frozen", "Each set's SHA-256 is stamped at freeze time and re-checked against the file "
                       "and against the digest the evaluation recorded using."),
        ("Stated limits", "Small sets — 10 + 18 + 20 questions. No statistical significance is "
                          "claimed anywhere in this project."),
    )),
)


def render(ctx: Context) -> None:
    meta = ctx.meta
    if meta:
        gen = meta.get("generation") or {}
        c.write('<div class="eyebrow">Live configuration</div><div style="height:12px"></div>')
        c.tile_row([
            c.metric_tile("Embedding", str(meta.get("model", "")).split("/")[-1],
                          f"{meta.get('dimensions')}-d · {meta.get('distance')}", "accent"),
            c.metric_tile("Indexed", f"{meta.get('chunk_count'):,}" if meta.get("chunk_count") else "—",
                          meta.get("collection", ""), "verified"),
            c.metric_tile("top-k", gen.get("top_k", "—"), "chunks requested"),
            c.metric_tile("Evidence floor", gen.get("score_threshold", "—"), "minimum similarity", "aorta"),
            c.metric_tile("LLM", str(gen.get("model") or "—").split("/")[-1],
                          "key configured" if gen.get("api_key_supplied") else "no key",
                          "verified" if gen.get("api_key_supplied") else "aorta"),
        ])

    for title, rows in SECTIONS:
        c.write(f'<hr class="hair"><div class="eyebrow">{c.esc(title)}</div>'
                '<div style="height:12px"></div>')
        c.write(c.panel("", c.definition_list(rows, prose_keys=tuple(k for k, _ in rows))))

    # -- provenance --------------------------------------------------------
    if meta:
        provenance = meta.get("index_provenance") or {}
        connection = meta.get("connection") or {}
        c.write('<hr class="hair"><div class="eyebrow">Index provenance</div>'
                '<div style="height:12px"></div>')
        c.write(c.panel("", c.definition_list([
            ("Model revision", meta.get("revision")),
            ("Vector store", meta.get("vector_store")),
            ("Connection mode", connection.get("mode")),
            ("Collection", meta.get("collection")),
            ("Distance", meta.get("distance")),
            ("Vectors", provenance.get("n_vectors")),
            ("Max chunk tokens", provenance.get("max_chunk_tokens")),
            ("Model token limit", provenance.get("token_limit")),
            ("Source chunk set", provenance.get("source_chunks_file")),
            ("Source SHA-256", provenance.get("source_chunks_sha256")),
            ("Indexed IDs SHA-256", provenance.get("indexed_chunk_ids_sha256")),
            ("Manifest SHA-256", provenance.get("index_meta_sha256")),
            ("Built (UTC)", provenance.get("built_at_utc")),
        ])))
        c.write('<div class="tiny" style="margin-top:10px;max-width:68ch">These digests identify the '
                'exact chunk set the running collection was built from. The API verifies the binding '
                'on load and refuses to serve an index that has drifted from its own chunk set.</div>')

        with st.expander("Raw /v1/meta payload", expanded=False):
            st.json(meta, expanded=False)

    c.write('<hr class="hair"><div class="eyebrow">Repository layout</div>'
            '<div style="height:12px"></div>')
    c.write(c.data_table(["Path", "What lives there"], [
        ["ingestion/", "PDF extraction, cleaning, chunking"],
        ["retrieval/", "the local index and its chunk↔vector binding check"],
        ["vectordb/", "Qdrant schema, migration, retriever"],
        ["generation/", "prompts, safety, threshold, provider, parser, validator, pipeline"],
        ["api/", "the FastAPI transport layer — no core logic"],
        ["ui/", "this Streamlit client — HTTP only, imports no pipeline"],
        ["eval/", "frozen question sets, every experiment run, integrity checks"],
        ["tests/", "the test suite"],
    ]))
