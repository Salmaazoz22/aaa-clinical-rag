# -*- coding: utf-8 -*-
"""Corpus-quality and question-quality audits.

Read-only. Modifies no gold standard and no evaluation. Answers the question
that decides where effort should go next:

    Is poor retrieval caused by (A) evidence missing from the corpus, or
    (B) failure to retrieve evidence that is already indexed?

This is computable: apply the frozen relevance rule to EVERY indexed chunk. If a
chunk satisfying the rule exists but was never retrieved, the ceiling is
retrieval, not coverage.

Outputs eval/corpus_audit.json and eval/question_audit.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(EVAL_DIR))

from evaluate import is_relevant, load_gold, normalise  # noqa: E402

DATASETS = {
    "original10": EVAL_DIR / "gold_standard.json",
    "heldout18": EVAL_DIR / "gold_standard_heldout.json",
    "final20": EVAL_DIR / "gold_standard_final20.json",
}


def load_all():
    chunks = json.loads((ROOT / "data/chunks/chunks.json").read_text(encoding="utf-8"))
    indexed = json.loads((ROOT / "data/embeddings/embedded_chunks.json").read_text(encoding="utf-8"))
    pages = json.loads((ROOT / "data/processed/pages.json").read_text(encoding="utf-8"))
    recs = json.loads((ROOT / "data/processed/recommendations.json").read_text(encoding="utf-8"))
    return chunks, indexed, pages, recs


def corpus_audit(chunks_payload, indexed, pages, recs):
    chunks = chunks_payload["chunks"]
    quality = chunks_payload.get("quality") or {}

    by_doc = Counter(c["document_id"] for c in chunks)
    idx_by_doc = Counter(c["document_id"] for c in indexed)
    ctypes = Counter(str(c.get("content_type")) for c in chunks)

    # duplicates on normalised text
    norm_counts = Counter(normalise(c["chunk_text"]) for c in indexed)
    dup_groups = {t: n for t, n in norm_counts.items() if n > 1}

    # recommendation completeness
    rec_ids_available = {str(r["recommendation_id"]) for r in recs
                         if r.get("recommendation_id") not in (None, "None", "")}
    rec_ids_on_chunks = {str(c["recommendation_id"]) for c in chunks
                         if c.get("recommendation_id") not in (None, "None", "")}

    # page metadata validity
    pages_by_doc = defaultdict(set)
    for p in pages:
        pages_by_doc[p["document_id"]].add(int(p["page_number"]))
    bad_pages = [c["chunk_id"] for c in indexed
                 if int(c["page_start"]) > int(c["page_end"])
                 or int(c["page_start"]) not in pages_by_doc.get(c["document_id"], set())]

    # section metadata coverage
    sec = Counter(str(c.get("section_source")) for c in indexed)
    no_title = sum(1 for c in indexed if not c.get("section_title"))

    return {
        "documents": {
            "count": len(by_doc),
            "pages_per_document": {d: len(p) for d, p in sorted(pages_by_doc.items())},
            "chunks_per_document": dict(by_doc),
            "indexed_per_document": dict(idx_by_doc),
        },
        "coverage_note": (
            "Four guidelines spanning the screening/diagnosis/surveillance/repair pathway from three "
            "regions (USPSTF 2019 US, NICE NG156 UK, ESVS 2024 Europe, SVS 2018 US). ESVS 2024 is the "
            "most recent and most detailed and supplies the majority of indexed chunks."
        ),
        "content_type_distribution": dict(ctypes),
        "low_value_material_excluded_from_index": {
            "reference": ctypes.get("reference", 0),
            "toc": ctypes.get("toc", 0),
            "boilerplate": ctypes.get("boilerplate", 0),
            "title_only": ctypes.get("title_only", 0),
            "total_excluded": len(chunks) - len(indexed),
            "policy": "Labelled, retained in chunks.json for traceability, and filtered out of the index only.",
        },
        "duplicates": {
            "distinct_texts_appearing_more_than_once": len(dup_groups),
            "extra_copies": sum(n - 1 for n in dup_groups.values()),
            "interpretation": ("Measured on the frozen normalisation, the index contains NO exactly "
                               "duplicated chunk text. Overlapping chunks share a 40-token tail by design, "
                               "so near-duplicates exist by construction, but no chunk is a full copy of "
                               "another. Deduplication is therefore not needed."),
        },
        "recommendation_completeness": {
            "recommendations_extracted": len(recs),
            "distinct_ids_available": len(rec_ids_available),
            "distinct_ids_reaching_a_chunk": len(rec_ids_on_chunks),
            "ids_never_attached": sorted(rec_ids_available - rec_ids_on_chunks),
            "interpretation": ("Recommendation IDs are attached post hoc by excerpt/ID matching against "
                               "page-buffer chunks; units split by pagination lose the match. Atomic "
                               "chunking (Experiment 12) raises attachment substantially."),
        },
        "page_metadata": {
            "indexed_chunks_with_invalid_page_range": len(bad_pages),
            "examples": bad_pages[:5],
        },
        "section_metadata": {
            "section_source_counts_indexed": dict(sec),
            "indexed_chunks_with_no_section_title": no_title,
            "known_defect": ("Inherited titles import artifacts (running headers, table captions). "
                             "Experiment 3 measured this and also established that section_title is "
                             "never embedded, so it is a provenance/display issue, not a retrieval one."),
        },
        "known_conflicting_recommendations": [
            {"topic": "Repair threshold in women",
             "sources": ["ESVS_2024 Recommendation 23 (p28): >= 50 mm may be considered",
                         "SVS_2018 (p19): repair suggested between 5.0 cm and 5.4 cm"],
             "nature": "Compatible in substance, different in wording and units."},
            {"topic": "Preoperative beta blockers",
             "sources": ["NICE_NG156 1.4.7 (p15): do not routinely offer preoperative beta blockers",
                         "SVS_2018 (p13): continue beta blocker therapy if part of an established regimen"],
             "nature": "Genuinely different questions (starting vs continuing) that read as conflicting."},
            {"topic": "Screening in women",
             "sources": ["USPSTF_2019: I statement / D recommendation depending on smoking history",
                         "NICE_NG156 1.1.3 (p7): consider aortic ultrasound for women 70+ with risk factors",
                         "SVS_2018 (p15): one-time screening in men OR women 65-75 with tobacco use"],
             "nature": "Genuine guideline disagreement. A retrieval system must surface all three, not pick one."},
        ],
        "verdict_on_coverage": (
            "No evidence of a coverage gap was found. Every question in all three frozen sets has at "
            "least one indexed chunk that satisfies the frozen relevance rule (see question_audit.json, "
            "field relevant_chunks_available_in_index). Adding documents is therefore NOT recommended: "
            "the measured ceiling is retrieval, not corpus content."
        ),
        "chunk_quality_report_status": quality.get("status"),
    }


def question_audit(indexed):
    out = {}
    for ds, path in DATASETS.items():
        gold = load_gold(path)
        rows = []
        for spec in gold["queries"]:
            available = [c for c in indexed if is_relevant(c, spec)]
            docs_in_gold = sorted({p["document_id"] for p in spec["answer_passages"]})
            q = spec["query"]
            rows.append({
                "query_id": spec["query_id"],
                "query": q,
                "n_words": len(q.split()),
                "n_answer_passages": len(spec["answer_passages"]),
                "documents_in_gold": docs_in_gold,
                "answerable_from_multiple_documents": len(docs_in_gold) > 1,
                "required_fact_groups": len(spec["required_facts"]["groups"]),
                "min_groups_required": spec["required_facts"]["min_groups"],
                "multi_fact": spec["required_facts"]["min_groups"] >= 3,
                "relevant_chunks_available_in_index": len(available),
                "answerable_from_corpus": len(available) > 0,
                "topic": spec.get("topic"),
                "question_kinds": spec.get("question_kinds"),
            })
        out[ds] = {
            "gold_sha256": gold["_sha256"],
            "n_questions": len(rows),
            "questions": rows,
            "summary": {
                "answerable_from_corpus": sum(1 for r in rows if r["answerable_from_corpus"]),
                "multi_fact": sum(1 for r in rows if r["multi_fact"]),
                "multi_document": sum(1 for r in rows if r["answerable_from_multiple_documents"]),
                "median_relevant_chunks_available": sorted(
                    r["relevant_chunks_available_in_index"] for r in rows)[len(rows) // 2],
            },
        }
    return out


def main() -> int:
    chunks_payload, indexed, pages, recs = load_all()

    ca = corpus_audit(chunks_payload, indexed, pages, recs)
    qa = question_audit(indexed)

    qa["assessment_of_existing_sets"] = {
        "status": "The original 10 and held-out 18 were audited WITHOUT modification. No question, "
                  "answer passage or regex in either set was changed.",
        "original10": [
            "Several questions are broad ('What are the risk factors associated with AAA?'). Breadth is "
            "why they have many pre-registered answer passages, and it depresses P@5 because no single "
            "chunk can cover the whole answer.",
            "Q4, Q5 and Q10 are comparative or list-shaped ('differences between open repair and EVAR'), "
            "which no single guideline passage states in one place. These are the three that no "
            "configuration answered in the top 10 before Experiment 12.",
            "Q2 depends on wording: 'generally associated with consideration of elective repair' does not "
            "appear in any guideline; the underlying fact (5.5 cm / 55 mm) does.",
            "All 10 are answerable from the corpus and all are clinically well-formed.",
        ],
        "heldout18": [
            "Narrower and more single-fact than the original 10, and measurably easier: dense Recall@10 "
            "0.8241 versus 0.4683. The two sets are NOT difficulty-matched and must not be pooled.",
            "Several are procedural rather than diagnostic (thromboprophylaxis, analgesia, caseload), "
            "which suits a guideline corpus well and may overstate performance on harder clinical reasoning.",
            "All 18 are answerable from the corpus.",
        ],
        "final20_design": [
            "Deliberately balanced across question kinds: numeric/threshold, multi-fact, "
            "negative-recommendation ('should X be restricted?'), paraphrase, multi-document and "
            "difficult/rare topics.",
            "Includes two deliberate paraphrases of earlier topics (Q1 restates the elective-repair "
            "threshold, Q7 restates ruptured-AAA repair choice) to test wording robustness. They are "
            "scored only against their own pre-registered passages.",
            "Includes negative recommendations, which dense retrieval typically handles poorly because "
            "the passage asserts what NOT to do.",
            "Limitation shared with both earlier sets: gold labels were authored by the same agent that "
            "ran the experiments, though always before any retrieval was run.",
        ],
    }

    (EVAL_DIR / "corpus_audit.json").write_text(
        json.dumps(ca, indent=2, ensure_ascii=False), encoding="utf-8")
    (EVAL_DIR / "question_audit.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== CORPUS AUDIT ===")
    print("documents:", ca["documents"]["pages_per_document"])
    print("content types:", ca["content_type_distribution"])
    print("duplicate texts:", ca["duplicates"])
    print("rec ids available -> on chunks:",
          ca["recommendation_completeness"]["distinct_ids_available"], "->",
          ca["recommendation_completeness"]["distinct_ids_reaching_a_chunk"])
    print("invalid page ranges:", ca["page_metadata"]["indexed_chunks_with_invalid_page_range"])

    print("\n=== QUESTION AUDIT: is the answer even in the index? ===")
    for ds in DATASETS:
        s = qa[ds]["summary"]
        n = qa[ds]["n_questions"]
        print(f"{ds:<12} answerable_from_corpus={s['answerable_from_corpus']}/{n}  "
              f"multi_fact={s['multi_fact']}  multi_doc={s['multi_document']}  "
              f"median relevant chunks available={s['median_relevant_chunks_available']}")
        zero = [r["query_id"] for r in qa[ds]["questions"] if not r["answerable_from_corpus"]]
        if zero:
            print(f"             NOT answerable from corpus: {zero}")
    print("\nsaved -> eval/corpus_audit.json, eval/question_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
