### Final decision: **ADOPT WITH CAVEATS** — V1 is the shipped chunker

**Shipped:** `V1_atomic_pagesafe` — structural-anchor chunk boundaries with recommendations kept
whole, page breaks still cutting narrative, under the existing hard token budget, with
`REJECT_CITATION_HEADINGS = True`.
**Rejected:** `V2_atomic_pure`.
**Unchanged:** the retriever. MedEmbed-base-v0.1 at pinned revision, dense cosine, top-10, no
reranking, no query processing.

#### Why V1

It is the only change in the project's history to raise P@1, and the only one to do so on a set
frozen **before the change existed**:

| set | P@1 base → V1 | MRR base → V1 | Recall@10 base → V1 | status of the set |
|---|---|---|---|---|
| original10 | 0.500 → **0.600** | 0.619 → **0.775** | 0.468 → **0.558** | used to select among variants |
| heldout18 | 0.556 → **0.722** | 0.697 → **0.815** | 0.824 → **0.898** | used to select among variants |
| **final20** | 0.400 → **0.550** | 0.531 → **0.664** | 0.608 → **0.783** | **pre-registered, frozen before the comparison** |

Both confounds were controlled, and V1 survives both:

- **Chunk size (V3).** Baseline algorithm, no anchors, enlarged budgets: P@1 moves 0.000 on both
  sets. Size is not the driver. (Partial control — it reached 199.8 mean tokens against V1's
  221.4, covering ~28% of the gap.)
- **Page-span overlap (V4).** Widening page metadata alone — identical vectors, identical
  ranking, identical retrieved chunks — is worth **P@1 +0.10 / +0.111**. Judging every chunk on
  its start page only removes that term entirely, and V1's P@1 is then **unchanged on all three
  sets** (0.600 / 0.722 / 0.550). V1's chunks average 1.23 pages against the baseline's 1.005, so
  it was barely exposed.

#### Why V2 was rejected despite equal or better headline numbers

V2 posts the highest raw metrics in the project (original10 P@1 0.700, MRR 0.820). Its chunks
average 5.57 pages and 63–81% of what it retrieves is multi-page. With the page term removed its
original10 P@1 collapses **0.700 → 0.400, below the 0.500 baseline**. Its true value is bounded
between those extremes and is **unresolved**. The configuration with the best-looking numbers had
the weakest evidence for them.

#### FINAL CORRECTED VALIDATION (the promotion gate)

Every historical V1/V2 number was produced with a known anchor defect present: numbered
*bibliography* lines were accepted as section headings (all 15 USPSTF "sections" were reference
entries such as `3 Svensjö S, Björck M, Gürtelschmid M, Djavani`). The fix was gated off so the
historical artifacts stayed reproducible. The gate re-ran V1 with the fix **on**, against
`final20` only, tuning nothing:

| metric | historical V1 (fix off) | corrected V1 (fix on) | Δ |
|---|---:|---:|---:|
| P@1 | 0.5500 | **0.5500** | +0.0000 |
| P@3 | 0.3333 | 0.3333 | +0.0000 |
| P@5 | 0.3000 | 0.3000 | +0.0000 |
| MRR | 0.6642 | **0.6642** | +0.0000 |
| Recall@5 | 0.6667 | 0.6667 | +0.0000 |
| Recall@10 | 0.7833 | 0.7833 | +0.0000 |
| Relevant Top-1 | 11/20 | 11/20 | +0.0000 |
| Answering@5 | 16/20 | 16/20 | +0.0000 |

**Every metric identical.** The fix removed 15 bogus USPSTF section anchors (15 → 0) and 13 bogus
ESVS ones (113 → 100); NICE and all recommendation anchors were untouched. Seven of twenty
questions retrieved a *differently identified* top-1 chunk — chunk IDs shift when boundaries move
— but the rank and relevance of every question were unchanged. The removed anchors only ever
fragmented bibliography text, which the content classifier excludes from the index anyway.

Artifact: `eval/runs/final_corrected_v1_final20.json`. It overwrites nothing.

#### The caveats, in full

1. **The mechanism reaches only half the corpus.** Anchors exist in ESVS 2024 (162 recommendation
   + 100 section) and NICE NG156 (57 + 14). **USPSTF 2019 and SVS 2018 yield none** — USPSTF's
   headings are unnumbered prose, SVS is a slide deck — and both fall back to page/token
   splitting.
2. **V3 is a partial size control** (~28% of the gap). The residual is not excluded.
3. **Small sets, single runs, no significance claimed.** One question is 0.05 of P@1 on final20.
4. **final20 has now been used.** A fourth frozen set is required before the next configuration
   decision; final20 can no longer arbitrate.
5. **Historical evaluations were scored against the previous index.** The original10 and
   heldout18 rows in `eval/final_evaluation_results.json`, and every run in `eval/runs/`, were
   produced with the page-buffer chunker, preserved in `data/archive_baseline_index/`. They
   remain valid historical records and were not recomputed.

#### Status

V1 is the **shipped default**: `clinical_chunking.DEFAULT_CHUNKER = "atomic"`, and
`data/chunks` + `data/embeddings` were rebuilt with it by `eval/rebuild_shipped_index.py`. The
historical page-buffer chunker is retained as `clinical_chunking.build_chunks` and reachable via
`run_chunking(strategy="page_buffer")`, so pre-Experiment-12 results stay reproducible.
