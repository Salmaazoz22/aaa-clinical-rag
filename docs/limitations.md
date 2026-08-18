# Limitations

Everything below is a limitation of the work as it actually stands. Nothing here is
hypothetical, and nothing that was verified is listed as a limitation.

---

## 1. Evaluation scale and statistics

- The three frozen sets contain **10, 18 and 20 questions**. On the original 10, one question is
  0.10 of P@1. **No statistical significance is claimed anywhere in this project**, and no
  confidence intervals are computed. Differences of one or two questions should be read as
  suggestive, not decisive.
- The three sets have **different difficulty and must never be pooled or averaged**. Dense
  Recall@10 is 0.4683 on the original 10 versus 0.8241 on the held-out 18; by relevant-chunks-
  available, final20 is the most demanding (median 4 relevant chunks per question, versus 7 for
  held-out 18 and 20 for the original 10).
- Gold labels for all three sets were **authored by the same agent that ran the experiments**.
  They were always written before retrieval was run against them, and final20 was additionally
  validated against source page text and hash-frozen — but no independent clinician reviewed them.
- **The `final20` freeze hash no longer matches the file it certifies.** The stamp taken at freeze
  time (`eval/gold_standard_final20.sha256`, `36af11b8…`) disagrees with the committed
  `eval/gold_standard_final20.json` (`5e02dbb8…`): the file was modified after the stamp was taken,
  and the original bytes exist nowhere in history. **The metrics are unaffected** — every published
  `final20` number reproduces exactly from the file as it stands — but the *audit trail* for the
  freeze is broken. It was deliberately **not** re-stamped: a fresh hash would assert a
  pre-registration property that can no longer be evidenced. `eval/integrity_report.json` records
  this as a FAIL rather than hiding it. Full detail: `docs/REFERENCE_COMPARISON.md` §9.

## 2. Test-set consumption

- Experiment 12 scored V1/V2/V3/V4 against **both** the original 10 and the held-out 18. After
  that point, **neither set is untouched with respect to the chunking decision**. This is why
  `final20` was authored and frozen before the final comparison was run.
- `final20` has now itself been used once. Any *future* configuration choice needs a fourth set.
  Selection consumes test sets; this must be budgeted in advance.

## 3. The relevance rule rewards two things that are not retrieval quality

Measured, not suspected:

- **Page-span width.** Relevance requires the chunk's page range to overlap the answer passage's
  page range. Experiment 12 V4 widened page metadata on the production index — *identical
  retrieval, identical ranking* — and gained **P@1 +0.10 (original 10) and +0.1111 (held-out 18)**.
  Any chunker producing wider page spans inherits this for free. It is why V2 was rejected
  despite having the best headline numbers in the entire project.
- **Chunk length.** A longer chunk satisfies more required-fact groups. V3 controlled for this
  only **partially**: it reached a 199.8-token mean against V1's 221.4 (baseline 191.6), because
  the page buffer flushes at page ends before an enlarged budget binds. It covers roughly 28% of
  the size gap. The residual is not ruled out.

The conservative "start-page-only" scoring in `eval/runs/exp12_atomic_chunking.json` removes the
page term entirely but is a strict **lower bound**: it penalises chunks that legitimately span
pages. For V1 (14.8% multi-page) the bound is tight; for V2 (81.6%) it is very loose, which is
exactly why V2's true value is unresolved.

## 4. Scope of the chunking improvement

- Structural anchors are found in only **two of the four documents**: ESVS 2024 (162
  recommendation + 100 section anchors) and NICE NG156 (57 + 14). **USPSTF 2019 and SVS 2018
  yield none** — USPSTF's headings are unnumbered prose and SVS is a slide deck. Both fall back
  to baseline page/token splitting, so the gain does not apply corpus-wide.
- Project B covers that gap only with hardcoded per-document header lists, which were refused
  because they must be hand-edited for every new guideline.
- **The anchor defect is fixed, scored, and immaterial — this is now resolved.** The first
  implementation accepted numbered *bibliography* lines as section headings (all 15 USPSTF
  "sections" were reference entries such as `3 Svensjö S, Björck M, Gürtelschmid M, Djavani`).
  The FINAL CORRECTED VALIDATION re-ran V1 with the fix on against `final20`:
  **all eight metrics identical, Δ 0.0000** (`eval/runs/final_corrected_v1_final20.json`). The
  removed anchors only ever fragmented bibliography text, which the content classifier already
  excludes from the index. The shipped chunker has the fix on.
  **Residual caveat, now closed:** the corrected configuration was originally validated on
  `final20` only, as instructed. `original10` and `heldout18` have since been recomputed on the
  shipped chunker (`eval/runs/p1_shipped_chunker_all_sets.json`): `original10` is identical in all
  eight metrics, and `heldout18` moved in three — P@3 +0.0185, P@5 +0.0112, MRR +0.0093 — all
  upward and all traceable to a single question whose first relevant hit moved from rank 3 to
  rank 2. The figures in `eval/final_evaluation_results.json` are preserved as the historical
  record; the README carries both the superseded and the recomputed `heldout18` row.

## 5. Selective reranking (Policy A) is unresolved, not adopted

Phase 7 was **INCONCLUSIVE** and is described that way everywhere. On the held-out 18 it
preserved all 10 dense-correct top-1 answers and added one, but:

- the gain is a single question (P@1 +0.0556 = 1/18);
- a **+0.0011 change to the frozen threshold reverses it**;
- the threshold was derived from the original 10 (transductive) and only frozen afterwards;
- the Q4 structural pattern that motivated it occurred in **0 of 18** held-out questions.

It has **not** been retuned, and it was **not** evaluated against final20 — deliberately, so that
final20 measures the chunking question alone. It remains experimental.

## 6. What the evaluation does not measure at all

- **Retrieval only.** There is no answer generation, no citation-faithfulness check, and no
  hallucination evaluation anywhere in this repository.
- **Fact coverage is regex-based**, not semantic. A chunk that states a fact in wording the
  pre-registered patterns do not anticipate is scored as irrelevant.
- **No clinical safety evaluation.** Nothing here establishes that surfacing these passages is
  safe for clinical decision-making, and the corpus contains genuinely **conflicting**
  recommendations across guidelines (repair threshold in women; preoperative beta blockers;
  screening in women) that the system surfaces without reconciling.

## 7. Engineering and deployment gaps

Verified in `eval/stability_report.json`:

- **No service exists.** No API, no authentication, no monitoring, no rate limiting.
- **No logging or observability** in the retrieval path.
- **No abstention.** An empty, punctuation-only or entirely out-of-scope query still returns 10
  chunks; there is no similarity floor below which the system declines to answer.
- **Historical evaluations were scored against a different index than the one that now ships.**
  This is the reproduction boundary, and it is deliberate rather than accidental. Promoting V1
  rebuilt `data/chunks` and `data/embeddings` (**1,760 chunks → 991 indexed**). Every run in
  `eval/runs/`, and the `original10` / `heldout18` / `final20` rows in
  `eval/final_evaluation_results.json`, were produced against the previous page-buffer index
  (2,116 → 1,330), preserved intact in `data/archive_baseline_index/`. Those results were **not**
  recomputed; they remain valid historical records of the comparison that drove the decision.
  The page-buffer chunker is retained as `clinical_chunking.build_chunks` and reachable via
  `run_chunking(strategy="page_buffer")`.
  The shipped artifacts **do** now reproduce from the shipped code — that stability check passes,
  which it did not before promotion.
- **Clean-checkout reproducibility from the source PDFs was NOT verified**, because doing so
  requires deleting the artifacts every preserved evaluation is scored against. Chunking
  determinism from `data/processed` and index reproducibility by re-embedding *were* verified;
  the unverified link is PDF → `data/processed`.
- The index is an **in-process NumPy matrix with exhaustive cosine search**. Correct and exactly
  reproducible at 1,330 vectors; it is not an ANN service and will not scale unchanged.
- Latency figures are **CPU-only** and dominated by query encoding.

## 8. Corpus limitations

- Four guidelines (USPSTF 2019, NICE NG156, ESVS 2024, SVS 2018). ESVS 2024 supplies about 82%
  of indexed chunks, so the corpus is heavily weighted toward one document.
- **No coverage gap was found**: all 48 questions across the three frozen sets have at least one
  indexed chunk satisfying the frozen relevance rule, so the measured ceiling is retrieval, not
  corpus content. Adding documents is *not* recommended on current evidence.
- **22 of 57** distinct recommendation IDs never reach a chunk under baseline chunking, because
  IDs are attached post hoc and pagination splits the units they belong to. Atomic chunking
  improves this (350–376 chunks carry an ID versus 172), but this is provenance, not retrieval.
- Section titles carry roughly 30% artifacts under the baseline (running headers, table
  captions). Experiment 3 established that `section_title` is **never embedded**, so this is a
  provenance and display defect, not a retrieval one.

## 9. Project B

- Project B was **not re-run**. Its reported P@1 0.40 → 0.90 figures were provided, not
  reproduced, and cannot be regenerated from its committed artifacts because it contains no
  metric harness. The source-level findings (intent firing 10/10 vs 1/18; 9 of 10 checker
  patterns reusing ranker literals; 143/452 chunks over the token window) **were** measured
  directly and stand independently.
- The claim that Project B's one held-out intent firing would be *harmful* is read from its
  rules, not measured.
