# -*- coding: utf-8 -*-
"""Generate notebooks/final_evaluation.ipynb.

The notebook reads only committed artifacts; it computes no metric of its own, so
it cannot disagree with eval/. Re-running this script regenerates the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "final_evaluation.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = []
A = CELLS.append

A(md("""# AAA Clinical RAG — Final Evaluation

Evidence-first evaluation of a retrieval system over four abdominal aortic aneurysm (AAA)
clinical guidelines.

**How to read this notebook.** Every number is loaded from a committed artifact under `eval/`.
Nothing is recomputed here, so this notebook cannot disagree with the evaluation that produced
those artifacts. Three frozen question sets are reported **separately and never pooled**.

Three words are used precisely throughout:

| term | meaning |
|---|---|
| **we tried it** | the experiment ran and is preserved |
| **it worked** | it improved metrics on a frozen set it was measured against |
| **it generalized** | it improved metrics on a set that was frozen *before* the change existed |

Very little in this project reaches the third category. That is the honest result."""))

A(code('''import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 220)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EVAL = ROOT / "eval"

def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))

RESULTS   = load("eval/final_evaluation_results.json")
EVIDENCE  = load("eval/final_evidence.json")
HISTORY   = load("eval/experiment_history.json")
CORPUS    = load("eval/corpus_audit.json")
QUESTIONS = load("eval/question_audit.json")
PROJECTB  = load("eval/project_b_comparison.json")
EXP12     = load("eval/runs/exp12_atomic_chunking.json")
INDEXMETA = load("data/embeddings/index_meta.json")
CHUNKS    = load("data/chunks/chunks.json")

try:
    STABILITY = load("eval/stability_report.json")
except FileNotFoundError:
    STABILITY = None

KEYS = ["P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5"]
DATASETS = ["original10", "heldout18", "final20"]

print("artifacts loaded")
for name, sha in RESULTS["gold_sha256"].items():
    print(f"  gold {name:<12} sha256 {sha}")'''))

# 1
A(md("""## 1. Project overview

**Task.** Given a clinical question about abdominal aortic aneurysm, retrieve the guideline
passages that answer it, with full provenance (document, page, section, chunk id).

**Design commitments**, held throughout:

- Retrieval is **dense cosine similarity only**. No query rewriting, no intent detection, no
  keyword bonuses, no per-question rules. Nothing in the retrieval path can branch on *which*
  question is being asked — a rule that cannot see the question cannot be fitted to it.
- Every change is scored by a **frozen, retriever-agnostic evaluator** against pre-registered
  answer passages. Reverting is the normal outcome.
- Low-value material (references, contents pages, boilerplate) is **labelled and excluded from
  the index, never deleted** from `chunks.json`."""))

# 2
A(md("## 2. Corpus overview"))
A(code('''docs = CORPUS["documents"]
print("pages per document :", docs["pages_per_document"])
print("total pages        :", sum(docs["pages_per_document"].values()))
print("chunks per document:", docs["chunks_per_document"])
print("indexed per doc    :", docs["indexed_per_document"])
print()
print(CORPUS["coverage_note"])'''))

# 3
A(md("""## 3. Guideline / document inventory

| id | guideline | year | region |
|---|---|---|---|
| `USPSTF_2019` | US Preventive Services Task Force — AAA screening recommendation statement | 2019 | US |
| `NICE_NG156` | NICE NG156 — Abdominal aortic aneurysm: diagnosis and management | 2020 | UK |
| `ESVS_2024` | ESVS 2024 Clinical Practice Guidelines — abdominal aorto-iliac artery aneurysms | 2024 | Europe |
| `SVS_2018` | Society for Vascular Surgery — AAA guideline (slide deck) | 2018 | US |

The corpus deliberately contains **genuinely conflicting recommendations**. A retrieval system
must surface all sides rather than silently pick one."""))
A(code('''for c in CORPUS["known_conflicting_recommendations"]:
    print(f"* {c['topic']}  [{c['nature']}]")
    for s in c["sources"]:
        print("    -", s)'''))

# 4
A(md("## 4. Chunk statistics"))
A(code('''q = CHUNKS["quality"]
print(f"total chunks   : {q['total']}")
print(f"valid          : {q['valid']}    invalid: {q['invalid']}    duplicates: {q['duplicates']}")
print(f"status         : {q['status']}")
print(f"content types  : {q['content_types']}")
print(f"tokens         : {q['tokens']}")
print()
print("excluded from the index (labelled, not deleted):")
print(" ", CORPUS["low_value_material_excluded_from_index"])'''))

# 5
A(md("""## 5. Old vs final chunking

Baseline chunking is **page-driven**: a full guideline page already exceeds `TARGET_CHARS`, so
the buffer flushes at nearly every page end and 2,109 of 2,116 chunks are single-page. A
recommendation split by a page break is split across two chunks, and recommendation identity is
recovered only post hoc.

Experiment 12 chunking is **structure-driven**: the document is cut at structural anchors
(`Recommendation N`, numbered recommendation IDs, numbered section headings) and a recommendation
stays whole, under the same hard token budget.

**V1 is now what ships** (`clinical_chunking.DEFAULT_CHUNKER = "atomic"`,
`notebooks/clinical_atomic_chunking.py`). The table below is the Experiment 12 comparison; the
live index built from the shipped chunker is reported in sections 4 and 7."""))
A(code('''prof = pd.DataFrame([{
    "variant": n,
    "chunks": v["chunk_profile"]["total_chunks"],
    "indexed": v["chunk_profile"]["indexed_chunks"],
    "mean tok": v["chunk_profile"]["tokens"]["mean"],
    "max tok": v["chunk_profile"]["tokens"]["max"],
    "over limit": v["chunk_profile"]["tokens"]["over_limit"],
    "pages/chunk": v["page_span_profile"]["mean_pages_per_chunk"],
    "% multipage": v["page_span_profile"]["pct_multi_page"],
    "with rec_id": v["chunk_profile"]["with_recommendation_id"],
} for n, v in EXP12["variants"].items()])
print(prof.to_string(index=False))'''))

# 6
A(md("""## 6. Token-budget validation

Chunk size is budgeted in **tokens**, measured with the tokenizer of the model that will actually
encode the text — not in characters. `embeddable_chunks` **raises** rather than letting an
oversized chunk be silently truncated at encode time.

This is the single clearest engineering difference from Project B, which enforces no token budget
and silently discards 83,516 tokens across 143 of its 452 indexed chunks."""))
A(code('''sys.path.insert(0, str(ROOT / "notebooks"))
import clinical_chunking as cc

idx = load("data/embeddings/embedded_chunks.json")
limit = INDEXMETA["token_limit"]
counts = [int(c["token_count"]) for c in idx]
print(f"model            : {INDEXMETA['model_name']}")
print(f"token limit      : {limit}")
print(f"max chunk tokens : {max(counts)}")
print(f"over the limit   : {sum(1 for t in counts if t > limit)}")
print()
b = PROJECTB["measured_project_b_facts"]
print("Project B, for contrast:")
print(f"  window {b['encoder_token_window']}, max chunk {b['indexed_token_max']}, "
      f"over window {b['indexed_chunks_over_token_window']}/{b['indexed_chunks']} "
      f"({b['percent_indexed_over_token_window']}%), "
      f"{b['tokens_silently_dropped_at_encode_time']} tokens dropped")'''))

# 7
A(md("## 7. Embedding / index statistics"))
A(code('''vecs = np.load(ROOT / "data/embeddings/embeddings.npy")
norms = np.linalg.norm(vecs, axis=1)
print(f"model        : {INDEXMETA['model_name']}")
print(f"revision     : {INDEXMETA['model_revision']}   (pinned by commit)")
print(f"dimensions   : {INDEXMETA['embedding_dim']}")
print(f"vectors      : {vecs.shape}  {vecs.dtype}")
print(f"normalisation: L2, min {norms.min():.8f} max {norms.max():.8f}")
print(f"index type   : {INDEXMETA['index_type']} / {INDEXMETA['metric']}")'''))

# 8
A(md("""## 8. Retrieval methodology

```
query -> MedEmbed-base-v0.1 (pinned revision) -> L2-normalised vector
      -> cosine against 1,330 normalised chunk vectors (exhaustive)
      -> top-10, with document / page / section / chunk_id provenance
```

There is no second stage in the production path. The optional cross-encoder
(`notebooks/clinical_rerank.py`) exists but is **not wired in**, and the selective reranking
policy is experimental — see section 12 and `docs/limitations.md`."""))

# 9
A(md("""## 9. Evaluation methodology

A retrieved chunk is **relevant** if and only if **both** hold:

1. **Provenance** — same `document_id`, and the chunk's `[page_start, page_end]` overlaps a
   pre-registered answer passage's page span.
2. **Facts** — the normalised chunk text satisfies at least `min_groups` of the query's
   `required_facts` groups.

No specific `chunk_id` is ever required, so the rule is independent of how the corpus is chunked.

**Three separate frozen sets.** They differ in difficulty and are never pooled."""))
A(code('''rows = []
for ds in DATASETS:
    qa = QUESTIONS[ds]
    s = qa["summary"]
    rows.append({
        "dataset": ds, "questions": qa["n_questions"],
        "gold sha256": RESULTS["gold_sha256"][ds][:16] + "...",
        "answerable from corpus": f"{s['answerable_from_corpus']}/{qa['n_questions']}",
        "multi-fact": s["multi_fact"], "multi-document": s["multi_document"],
        "median relevant chunks in index": s["median_relevant_chunks_available"],
    })
print(pd.DataFrame(rows).to_string(index=False))
print()
print("Every question in all three sets is answerable from the index, so the measured")
print("ceiling is RETRIEVAL, not corpus coverage. final20 is the hardest by available evidence.")'''))

# 10-12
A(md("""## 10-12. Results on the three frozen sets

> **Historical labels.** These three tables were produced **before** V1 was promoted.
> `baseline_production` is the *old* page-buffer chunker, preserved in
> `data/archive_baseline_index/`; `V1_atomic_pagesafe` is what now ships (measured here with the
> citation-heading fix off — section 17a shows it makes no difference). They are kept exactly as
> produced, so the comparison that drove the decision stays reproducible.

**The retriever is identical in all three rows** — same model, same pinned revision, same dense
cosine, no reranking. The only variable is where chunk boundaries fall."""))
A(code('''def table(ds):
    base = RESULTS["metrics"][ds]["baseline_production"]["metrics"]
    df = pd.DataFrame([{"config": c, **{k: m["metrics"][k] for k in KEYS}}
                       for c, m in RESULTS["metrics"][ds].items()])
    d = pd.DataFrame([{"config": "delta " + c,
                       **{k: round(m["metrics"][k] - base[k], 4) for k in KEYS}}
                      for c, m in RESULTS["metrics"][ds].items() if c != "baseline_production"])
    print(f"\\n=== {ds} ===")
    print(df.to_string(index=False))
    print()
    print(d.to_string(index=False))

for ds in DATASETS:
    table(ds)'''))

A(md("""### Reading these three tables

`final20` is the only set that was **frozen before the configuration being tested existed**.
The original 10 and held-out 18 were both used to score V1/V2/V3/V4 during Experiment 12, so with
respect to the chunking decision they are selection sets, not held-out sets. Where the three
tables agree, the result generalized. Where they disagree, final20 is the one to believe."""))

# 13
A(md("## 13. Complete experiment history"))
A(code('''hist = pd.DataFrame([{
    "#": e["n"], "phase": e["phase"], "experiment": e["name"][:52], "decision": e["decision"][:34],
} for e in HISTORY["experiments"]])
print(hist.to_string(index=False))'''))
A(code('''# Full record for any single experiment.
def show(n):
    e = next(x for x in HISTORY["experiments"] if x["n"] == n)
    for f in ["name", "phase", "problem", "hypothesis", "intervention", "dataset",
              "evidence", "result", "failure_mode", "decision", "lesson_learned"]:
        print(f"{f.upper():<16} {e[f]}")
    print(f"{'METRICS':<16}")
    for k, v in e["metrics"].items():
        print(f"                 {k}: {v}")

show(21)   # the page-span control -- the experiment that changed the conclusion'''))

# 14-15
A(md("""## 14-15. Per-query evidence inspection

For every question, in every dataset, for every configuration: what was retrieved, at what rank,
with what similarity, from which document / page / section / chunk, whether the frozen rule
scored it relevant, which pre-registered answer passages it matched, and which required-fact
groups its text covers.

This table is sufficient to **reconstruct every metric** reported above."""))
A(code('''EV = pd.DataFrame(EVIDENCE["rows"])
print(f"evidence rows: {len(EV)}  "
      f"({EV['dataset'].nunique()} datasets x {EV['config'].nunique()} configs x 10 ranks)")

def evidence_for(dataset, config, query_id, top=5):
    sel = EV[(EV.dataset == dataset) & (EV.config == config) & (EV.query_id == query_id)]
    print(f"\\nQ{query_id} [{dataset} / {config}]")
    print(sel.iloc[0]["question"])
    cols = ["rank", "similarity", "document_id", "page_start", "page_end",
            "section_title", "relevant", "required_facts_covered", "chunk_id"]
    print(sel[cols].head(top).to_string(index=False))

evidence_for("final20", "baseline_production", 1)
evidence_for("final20", "V1_atomic_pagesafe", 1)'''))
A(code('''# Expected evidence (pre-registered) vs what was actually retrieved, for one question.
gold20 = load("eval/gold_standard_final20.json")
spec = gold20["queries"][0]
print("QUESTION:", spec["query"])
print("\\nEXPECTED EVIDENCE (pre-registered before any retrieval):")
for p in spec["answer_passages"]:
    print(f"  {p['document_id']} p{p['page_start']}-{p['page_end']}  [{p['section_ref']}]")
    print(f"     why: {p['why']}")
print("\\nREQUIRED FACTS  (need >= "
      f"{spec['required_facts']['min_groups']} of {len(spec['required_facts']['groups'])} groups):")
for g in spec["required_facts"]["groups"]:
    print(f"  {g['name']}: {g['any_of']}")'''))

# 16
A(md("## 16. Failure analysis"))
A(code('''rows = []
for ds in DATASETS:
    for cfg, m in RESULTS["metrics"][ds].items():
        for q in m["per_query"]:
            if q["first_relevant_rank"] is None:
                rows.append({"dataset": ds, "config": cfg, "query_id": q["query_id"],
                             "n_answer_passages": q["n_answer_passages"],
                             "top1_doc": q["top1_doc"]})
fail = pd.DataFrame(rows)
if len(fail):
    print("Questions with NO relevant chunk anywhere in the top 10:\\n")
    print(fail.to_string(index=False))
    print()
    print("Counts by dataset/config:")
    print(fail.groupby(["dataset", "config"]).size().to_string())
else:
    print("No question failed completely in any dataset/configuration.")'''))
A(code('''# Which questions did the chunking change RESCUE, and which did it BREAK?
for ds in DATASETS:
    base = {q["query_id"]: q for q in RESULTS["metrics"][ds]["baseline_production"]["per_query"]}
    for cfg in [c for c in RESULTS["metrics"][ds] if c != "baseline_production"]:
        cur = {q["query_id"]: q for q in RESULTS["metrics"][ds][cfg]["per_query"]}
        gained = [i for i in base if cur[i]["relevant_top1"] and not base[i]["relevant_top1"]]
        lost   = [i for i in base if base[i]["relevant_top1"] and not cur[i]["relevant_top1"]]
        print(f"{ds:<12} {cfg:<22} top-1 gained {gained}   top-1 LOST {lost}")'''))

# 17
A(md("""## 17. Chunking experiments — and the two controls that decide them

Two things can raise these metrics **without any retrieval improvement at all**: the relevance
rule rewards chunks that cover more text, and it requires page-range overlap. Both were
controlled.

- **V3 (size control)** — baseline algorithm, *no anchors*, budgets enlarged toward V1's mean.
- **V4 (page-span control)** — production index, *identical ranking and identical retrieved
  chunks*, page ranges widened to V2's mean span. Anything it gains is pure measurement artifact.
- **start-page-only scoring** — every chunk judged on its first page only. This removes the page
  term entirely and is a strict lower bound."""))
A(code('''def exp12(key, title):
    base = EXP12["variants"]["control_production"][key]["metrics"]
    df = pd.DataFrame([{"variant": n, **{k: v[key]["metrics"][k] for k in KEYS}}
                       for n, v in EXP12["variants"].items()])
    print(f"\\n=== {title} ===")
    print(df.to_string(index=False))

exp12("original_10", "original 10")
exp12("heldout_18", "held-out 18")
exp12("original_10_startpage_only", "original 10 - START PAGE ONLY (page confound removed)")
exp12("heldout_18_startpage_only", "held-out 18 - START PAGE ONLY (page confound removed)")'''))
A(md("""**What the controls established.**

- **V4**: widening page metadata *alone*, with identical retrieval, gains **P@1 +0.10** (original
  10) and **+0.1111** (held-out 18). The confound is real and large.
- **V1** has a mean page span of 1.23 (baseline 1.005), so it is barely exposed. Its gains are
  **unchanged** under start-page-only scoring — the confound-free comparison.
- **V2** has a mean page span of 5.57 and 81.6% multi-page. Under start-page-only scoring its
  original-10 P@1 collapses from 0.70 to 0.40, *below* the 0.50 baseline. **The configuration
  with the best headline numbers has the weakest evidence.**
- **V3** is a *partial* size control: it reached a 199.8-token mean against V1's 221.4 (baseline
  191.6), covering roughly 28% of the size gap. The residual is not ruled out."""))

# 17a
A(md("""### 17a. FINAL CORRECTED VALIDATION — the promotion gate

Every V1/V2 number above was produced with a known anchor defect present: numbered *bibliography*
lines (`3 Svensjö S, Björck M, Gürtelschmid M, Djavani`) were being accepted as section headings.
The fix was gated off so the historical artifacts stayed reproducible.

The gate re-ran V1 with the fix **on**, against `final20` only, tuning nothing."""))
A(code('''CORR = load("eval/runs/final_corrected_v1_final20.json")

cmp = pd.DataFrame([{
    "metric": k,
    "historical V1 (fix off)": CORR["metrics_historical_v1_fix_off"][k],
    "corrected V1 (fix ON, SHIPPED)": CORR["metrics_corrected"][k],
    "delta": CORR["delta"][k],
} for k in KEYS])
print(cmp.to_string(index=False))

print("\\nanchors per document (without fix -> with fix):")
for doc in sorted(CORR["anchor_census"]["with_fix"]):
    w = CORR["anchor_census"]["with_fix"][doc]
    o = CORR["anchor_census"]["without_fix"][doc]
    print(f"  {doc:<14} sections {o.get('section',0):>4} -> {w.get('section',0):<4}"
          f"  recommendations {o.get('recommendation',0):>4} -> {w.get('recommendation',0)}")

hp, cp = CORR["chunk_profile_historical_v1"], CORR["chunk_profile_corrected"]
print(f"\\n{'chunk stat':<26}{'historical':>12}{'corrected':>12}")
for lbl, key in [("total chunks","total_chunks"), ("indexed chunks","indexed_chunks"),
                 ("with recommendation_id","with_recommendation_id")]:
    print(f"{lbl:<26}{hp[key]:>12}{cp[key]:>12}")
for lbl, key in [("mean tokens","mean"), ("max tokens","max"), ("over model limit","over_limit")]:
    print(f"{lbl:<26}{hp['tokens'][key]:>12}{cp['tokens'][key]:>12}")

print(f"\\nquestions whose rank/relevance changed: "
      f"{sum(1 for c in CORR['questions_with_any_change'] if c['historical_first_relevant_rank'] != c['corrected_first_relevant_rank'] or c['historical_relevant_top1'] != c['corrected_relevant_top1'])}")
print(f"questions whose top-1 CHUNK ID changed:  {CORR['n_questions_changed']}")
print(f"\\nDECISION: {CORR['decision']}")'''))
A(md("""**All eight metrics identical (Δ 0.0000).** The fix removed 15 bogus USPSTF section anchors
(15 → 0) and 13 bogus ESVS ones (113 → 100); NICE and every recommendation anchor were untouched.
Seven of twenty questions returned a differently *identified* top-1 chunk — chunk IDs shift when
boundaries move — with unchanged rank and relevance. The removed anchors only ever fragmented
bibliography text, which the content classifier already excludes from the index.

A defect can be real, worth fixing, and still immaterial. Measuring exactly zero change is a
stronger statement than never having found it."""))

# 18
A(md("## 18. Retrieval-model comparison (all completed configurations)"))
A(code('''hist_rows = []
for e in HISTORY["experiments"]:
    for ds, m in e["metrics"].items():
        if isinstance(m, dict):
            hist_rows.append({"experiment": e["name"][:46], "dataset/config": ds, **m})
        else:
            hist_rows.append({"experiment": e["name"][:46], "dataset/config": ds,
                              **{k: "n/a" for k in KEYS}})
comp = pd.DataFrame(hist_rows)
print(comp.to_string(index=False))
print()
print('"n/a" means: ' + HISTORY["unavailable_marker"])'''))

# 19
A(md("## 19. Stability and reproducibility"))
A(code('''if STABILITY is None:
    print("eval/stability_report.json not present -- run: python eval/run_stability_checks.py")
else:
    print("summary:", STABILITY["summary"])
    st = pd.DataFrame([{"check": c["check"][:66], "category": c["category"],
                        "status": c["status"], "sec": c["seconds"]}
                       for c in STABILITY["checks"]])
    print()
    print(st.to_string(index=False))'''))

# 20
A(md("## 20. Runtime / performance"))
A(code('''if STABILITY:
    for c in STABILITY["checks"]:
        if c["category"] == "performance":
            print(c["check"])
            for k, v in (c["detail"] or {}).items():
                print(f"   {k}: {v}")
            print()'''))

# 21
A(md("""## 21. Limitations

Full list: `docs/limitations.md`. The ones that most affect how these results should be read:

1. **10 / 18 / 20 questions.** One question is 0.10 of P@1 on the original 10. No significance
   is claimed anywhere.
2. **The relevance rule rewards page-span width** — measured at roughly +0.10 P@1 for free (V4).
3. **V3 controls chunk size only partially** (~28% of the gap).
4. **Anchors exist in only 2 of the 4 documents**; USPSTF and SVS fall back to baseline splitting.
5. **A known anchor defect (bibliography lines accepted as headings) is fixed in code but was
   present in every V1/V2 number here**, gated off so the artifacts still reproduce.
6. **Retrieval only** — no answer generation, no citation-faithfulness, no clinical safety
   evaluation. The corpus contains conflicting recommendations that the system surfaces
   without reconciling.
7. **Selective reranking Policy A remains INCONCLUSIVE** and was deliberately excluded from
   final20 so that final20 measures the chunking question alone."""))

# 22
A(md("## 22. Final recommendation"))
A(code('''print(HISTORY.get("final_recommendation", "See docs/experiment_history.md"))'''))

# 23
A(md("""## 23. Deployment readiness

See `docs/deployment_readiness.md` and `eval/stability_report.json`.

**Verified:** deterministic chunking, deterministic and unique chunk IDs, a pinned embedding
revision, index/metadata alignment with L2-normalised vectors, index reproducibility by
re-embedding, token safety with a fail-loud validator, robustness to malformed and adversarial
queries, ranking determinism, latency and footprint, and the test suite.

**Not production ready, and not claimed to be.** The repository ships no service, no API, no
logging, no authentication, no monitoring, and **no abstention threshold** — an empty or wholly
out-of-scope query still returns 10 chunks. Clean-checkout reproducibility *from the source PDFs*
was not verified, because that would destroy the artifacts every preserved evaluation is scored
against.

It is a reproducible research pipeline, which is what it was built to be."""))
A(code('''if STABILITY:
    print("production_ready:", STABILITY["production_ready"])
    print()
    print(STABILITY["production_readiness_statement"])'''))


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {NB} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
