# Retrieval Optimization Experiments

> **Fresh evaluation baseline.** These metrics were newly established for the current
> repository because the previous evaluation harness / gold standard was not present. They are
> **not** numerically comparable to any earlier audit.

Gold standard: `eval/gold_standard.json`, SHA-256
`0b8a443b69960bc5ac20311f0010926a2f131bbb5531ccf369f321f59ed2e5c1` — **identical for every
run below**. Methodology: `eval/README.md`. Baseline snapshot: `docs/BASELINE_SNAPSHOT.md`.

Every experiment started from the verified baseline, applied **one** change, rebuilt chunks and
embeddings, ran the full test suite, and was scored by the same frozen evaluator. Saved runs
live in `eval/runs/`.

---

## Summary

| # | Experiment | Decision |
|---|---|---|
| 1 | Page-spanning clinical recommendations | **REVERT** |
| 2 | Conservative header/footer cleaning | **REVERT** |
| 3 | Improved section-title detection | **REVERT** |
| 4 | Optional dense + cross-encoder reranker | **KEPT AS OPT-IN, NOT DEFAULT** |
| 5 | Embedding model swap → `BAAI/bge-base-en-v1.5` | **NOT DEFAULT** (validated optional config) |
| 6 | BGE-base top-30 → cross-encoder rerank → top-10 | **DO NOT MAKE DEFAULT** (hypothesis refuted) |
| 7 | Biomedical encoder → `abhinand/MedEmbed-base-v0.1` | **KEPT — now the production default** (adopted, reproduces Experiment 7 exactly) |

**Final state: the original baseline.** No change was made to chunking, embedding or default
retrieval. Nothing in the pipeline was modified.

### Fresh baseline vs every experiment

| Metric | Baseline | Exp 1 page-span | Exp 2 furniture | Exp 3 sections | Exp 4 rerank (30) | Exp 4 rerank (20) | Exp 5 BGE-base |
|---|---:|---:|---:|---:|---:|---:|---:|
| **P@1** | **0.5000** | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4000 |
| **P@3** | **0.3667** | 0.3667 | 0.3333 | 0.3667 | 0.3333 | 0.3000 | 0.4000 |
| **P@5** | **0.2800** | 0.2800 | 0.2800 | 0.2800 | 0.2400 | 0.2200 | 0.4000 |
| **MRR** | **0.5625** | 0.5611 | 0.5625 | 0.5611 | 0.5736 | 0.5500 | 0.6167 |
| **Recall@5** | **0.2217** | 0.2217 | 0.2217 | 0.2217 | 0.1817 | 0.2067 | 0.3250 |
| **Recall@10** | **0.2667** | 0.2667 | 0.2917 | 0.2667 | 0.2917 | 0.2717 | 0.4483 |
| **Relevant Top-1** | **5/10** | 5/10 | 5/10 | 5/10 | 5/10 | 5/10 | 4/10 |
| **Answering@5** | **6/10** | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 10/10 |
| queries improved | — | 0 | 2 | 0 | 4 | 3 | 6 |
| queries regressed | — | 1 | 1 | 1 | 3 | 4 | 4 |
| tests | 28/28 | 28/28 | 28/28 | 28/28 | 28/28 | 28/28 | 28/28 |

No experiment raised P@1 or Relevant Top-1. The only one that moved either moved them **down**
(Exp 5), while lifting every other metric substantially — see Experiment 5 for why that is not
enough to change the default.

---

## FRESH BASELINE

| Field | Value |
|---|---|
| Source PDFs | 4 |
| Pages | 249 |
| Chunks | 2,116 |
| Indexed chunks | 1,330 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dims | 384 |
| Max chunk tokens | 254 |
| Chunks over 256-token limit | 0 |
| Index/chunk alignment | 1,330 = 1,330 ✅ |
| Tests | 28/28 passing |

Retrieval: dense cosine over an L2-normalised NumPy matrix, top-10.

| Metric | Value |
|---|---:|
| P@1 | 0.5000 |
| P@3 | 0.3667 |
| P@5 | 0.2800 |
| MRR | 0.5625 |
| Recall@5 | 0.2217 |
| Recall@10 | 0.2667 |
| Relevant Top-1 | 5/10 |
| Answering@5 | 6/10 |

Per query (`eval/runs/fresh_baseline_dense.json`):

| Q | Query | Baseline top-1 | First relevant rank | Answering@5 |
|---|---|---|---:|---|
| 1 | Screening recommendations | ESVS p19 · *Recommendation 9* | 8 | no |
| 2 | Diameter for elective repair | ESVS p26 · *ABDOMINAL AORTIC ANEURYSM* | 1 | yes |
| 3 | Surveillance for small AAA | ESVS p22 · *ABDOMINAL AORTIC ANEURYSM* | 1 | yes |
| 4 | Indications for EVAR | ESVS p83 · *(none)* | — | no |
| 5 | Risk factors for AAA | ESVS p95 · *INFORMATION FOR PATIENTS* | — | no |
| 6 | Imaging modality | ESVS p21 · *ABDOMINAL AORTIC ANEURYSM* | 1 | yes |
| 7 | Rupture risk factors | USPSTF p2 · *Summary of Recommendations* | 1 | yes |
| 8 | Smoking cessation | NICE p11 · *Reducing the risk of rupture* | 1 | yes |
| 9 | Women and screening | USPSTF p3 · *Treatment* | 2 | yes |
| 10 | Open repair vs EVAR | NICE p40 · *Repairing unruptured aneurysms* | — | no |

Reproducibility was verified before any experiment: rebuilding chunks and embeddings from
unmodified code reproduced the baseline **exactly** — identical chunk IDs, text, page ranges and
token counts, and a maximum absolute embedding difference of **0.0**. Every comparison below is
therefore attributable to the change under test, not to build noise.

---

## EXPERIMENT 1 — Page-spanning recommendations

**Baseline:** fresh baseline (verified).
**Change:** bridge a page boundary when a clinical unit is cut in half by it.
**Files modified:** `notebooks/clinical_chunking.py` only.

### What the audit claim actually looks like

Confirmed: of 2,116 baseline chunks, **2,109 are single-page** — chunking is effectively
page-scoped, because a full guideline page exceeds `TARGET_CHARS` on its own and flushes at
every page end.

A boundary was treated as a genuine cut only when **all** of the following held: the page stops
without terminal punctuation (after discarding its footer), recommendation language appears
within the last 300 characters, the next page resumes mid-sentence (after discarding its
header), and the section title does not change. That selects **6 of 245 boundaries** — whole
pages are never merged merely for being adjacent. The 6:

| Boundary | What was being cut |
|---|---|
| ESVS p11→12 | "Women specific / recommendations are given whenever possible…" |
| ESVS p95→96 | "…should be offered a one time ultrasound screening examination / of their tummy…" |
| NICE p9→10 | recommendation 1.1.13 bullet list |
| NICE p14→15 | recommendation 1.4.4 |
| NICE p19→20 | **recommendation 1.6.1** "Consider EVAR or open surgical repair for people / with a ruptured infrarenal AAA" |
| NICE p35→36 | committee rationale sentence |

Bridged pages are welded with a single space (not a paragraph break) so the sentence rejoins,
and the running header/footer sitting between the halves is dropped **at the join only**.
`split_text` then re-splits under the same token budget, so no chunk grows.

### Result

| Field | Before | After |
|---|---:|---:|
| Chunks | 2,116 | 2,113 |
| Indexed chunks | 1,330 | 1,329 |
| Max tokens | 254 | 254 |
| Chunks over limit | 0 | 0 |
| Index/chunk alignment | valid | valid |
| Tests | 28/28 | 28/28 |

| Metric | Baseline | Exp 1 | Δ |
|---|---:|---:|---:|
| P@1 | 0.5000 | 0.5000 | 0 |
| P@3 | 0.3667 | 0.3667 | 0 |
| P@5 | 0.2800 | 0.2800 | 0 |
| MRR | 0.5625 | 0.5611 | **−0.0014** |
| Recall@5 | 0.2217 | 0.2217 | 0 |
| Recall@10 | 0.2667 | 0.2667 | 0 |
| Relevant Top-1 | 5 | 5 | 0 |

**Improved queries:** none. **Regressed:** Q1 (first relevant rank 8 → 9). **Unchanged:** 9.

**Decision: REVERT.**
**Reason:** MRR decreased, no metric improved, and one query regressed. The change fixes a real
textual defect — NICE 1.6.1 genuinely is severed mid-sentence — but the frozen evaluation shows
no retrieval benefit, and the rule is that evidence decides. A side effect also argued against
it: because a flush stamps one page range across all its pieces, bridging widened `page_start`–
`page_end` on 64 chunks that came wholly from a single page, which is a small loss of page
precision for no measured gain.

---

## EXPERIMENT 2 — Header/footer cleaning

**Baseline:** fresh baseline (re-verified bit-for-bit before starting).
**Change:** detect and remove repeated page furniture before chunking.
**Files modified:** `notebooks/clinical_chunking.py` only.

### Why a frequency test alone is unsafe

A naive "remove lines that recur on many pages" rule would have deleted, from ESVS alone:
`Recommendation 9` (66 pages), `Class` (66), `Level` (66), `ToE` (59), `IIa` (38) — the
recommendation headings and the entire evidence-grading apparatus — plus NICE's `1.5.3`
recommendation numbering (17 pages) and SVS's `Recommendation` table heading (32 pages). That is
exactly the destruction the brief forbids.

The rule actually used requires **three** conditions: the line shape recurs on ≥25% of the
document's pages (digits masked, so "Page 17 of 53" and "Page 18 of 53" collapse); **and** it is
either edge-anchored (≥90% of its occurrences in the first 2 or last 3 lines of a page) or
matches an explicit publisher-furniture pattern; **and** it carries no clinical signal —
recommendation language, a measurement, a numbered item, or the word "recommendation" all veto
removal.

### What was removed — 255 line occurrences, 12,800 characters

| Document | Shapes | Occurrences | Chars removed | Examples |
|---|---:|---:|---|---|
| NICE NG156 | 3 | 157 | 10,362 (**11.19%**) | `© NICE 2025. All rights reserved…`, `Abdominal aortic aneurysm: diagnosis and management (NG156)`, `Page N of` |
| ESVS 2024 | 1 | 69 | 1,656 (0.18%) | `Anders Wanhainen et al.` |
| USPSTF 2019 | 6 | 29 | 782 (1.43%) | `JAMA`, `Volume 322, Number 22`, `(Reprinted) jama.com`, `Clinical Review & Education…` |
| SVS 2018 | 0 | 0 | 0 | — (its `Recommendation` heading was correctly protected) |

**Safety verification: 0 measurements lost, 0 clinical-signal lines lost, 0 numeric deltas on
any page.** Page and source metadata unchanged.

### Result

| Field | Before | After |
|---|---:|---:|
| Chunks | 2,116 | 2,091 |
| Indexed chunks | 1,330 | 1,316 |
| Max tokens | 254 | 254 |
| Chunks over limit | 0 | 0 |
| Index/chunk alignment | valid | valid |
| Tests | 28/28 | 28/28 |

| Metric | Baseline | Exp 2 | Δ |
|---|---:|---:|---:|
| P@1 | 0.5000 | 0.5000 | 0 |
| P@3 | 0.3667 | 0.3333 | **−0.0334** |
| P@5 | 0.2800 | 0.2800 | 0 |
| MRR | 0.5625 | 0.5625 | 0 |
| Recall@5 | 0.2217 | 0.2217 | 0 |
| Recall@10 | 0.2667 | 0.2917 | **+0.0250** |
| Relevant Top-1 | 5 | 5 | 0 |

**Improved:** Q3 (P@5 0.20→0.40), Q6 (Recall@10 0.25→0.50). **Regressed:** Q9 (P@5 0.60→0.40).
**Unchanged:** 7.

**Decision: REVERT.**
**Reason:** every priority metric — P@1, MRR, Recall@5 — is exactly flat; only Recall@10, the
lowest-priority metric, moved (+0.025, one passage), and P@3 moved the other way. Net +1 query.
Under "if metrics are essentially unchanged, REVERT" and "if results are mixed, REVERT unless
the evidence is strong", this does not clear the bar.

**Noted for the record:** this is the strongest of the four candidates and the only one that
improved a metric without touching P@1/MRR. It is also correct on its own terms — 11% of NICE's
extracted characters are legal boilerplate that no reader wants retrieved. If the decision rule
were ever relaxed to "no regression on the priority metrics", this is the change to revisit
first — so its full implementation is reproduced in the appendix below rather than lost with the
revert. It is **not** present in the repository as executable code.

---

## EXPERIMENT 3 — Section title detection

**Baseline:** fresh baseline (re-verified bit-for-bit before starting).
**Change:** prefer a heading found inside the chunk; reject inherited titles that fail a heading
shape test (→ `None` rather than a guess).
**Files modified:** `notebooks/clinical_chunking.py` only.

### The defect is real

Baseline indexed chunks carried these as "section titles":

| Bogus title | Chunks | What it actually is |
|---|---:|---|
| `ABDOMINAL AORTIC ANEURYSM` | 133 | ESVS running header |
| `Table 1-continued` | 117 | table caption |
| `EVAR SIZING AND PLANNING WORKSHEET` | 71 | worksheet running head |
| `ABDOMINAL AORTIC ANEURYSM REPAIR` | 64 | running header |
| `Complex AAAs are estimated to constitute about 15 e` | 57 | a truncated body sentence |
| `4 RCT, 12 CS` | 27 | a table cell |
| `J Vasc Surg. 2018 Jan;67(1):2–77.e2` | 1 | a citation |

Detection added: `Recommendation N` headings, and numbered ESVS/NICE headings
(`1.6 Repairing ruptured aneurysms`) with a chapter cap, word cap, and rejection of author
lists, citations, dangling prose and table captions — so bibliography lines such as
`1261 Robertson L. Optimising intervals…` cannot masquerade as headings.

### Result — precision up, coverage down

Detected titles rose 451 → 459 with the artifacts above eliminated, but titles that failed the
shape test became `None`, so coverage fell:

| | Baseline | Exp 3 |
|---|---:|---:|
| Indexed chunks with a section title | 1,157 / 1,330 (87.0%) | 747 / 1,331 (56.1%) |
| `section_source = detected` | 451 | 459 |
| `section_source = inherited` | 706 | 288 |
| `section_source = unknown` (None) | 173 | 584 |

| Field | Before | After |
|---|---:|---:|
| Chunks | 2,116 | 2,116 |
| Indexed chunks | 1,330 | 1,331 |
| Max tokens | 254 | 254 |
| Chunks over limit | 0 | 0 |
| Tests | 28/28 | 28/28 |

| Metric | Baseline | Exp 3 | Δ |
|---|---:|---:|---:|
| P@1 | 0.5000 | 0.5000 | 0 |
| P@3 | 0.3667 | 0.3667 | 0 |
| P@5 | 0.2800 | 0.2800 | 0 |
| MRR | 0.5625 | 0.5611 | **−0.0014** |
| Recall@5 | 0.2217 | 0.2217 | 0 |
| Recall@10 | 0.2667 | 0.2667 | 0 |
| Relevant Top-1 | 5 | 5 | 0 |

**Improved:** none. **Regressed:** Q1 (rank 8 → 9). **Unchanged:** 9.

**Decision: REVERT.**
**Reason:** MRR decreased and nothing improved.

**Structural finding worth recording:** `section_title` is **metadata only — it is never
embedded**. `build_embeddings` encodes `chunk_text` alone. The only path by which a section title
can influence retrieval is `classify_chunk_content`, which consults it to spot reference
sections; here exactly one chunk changed classification (1,330 → 1,331 indexed), and that single
extra chunk is what pushed Q1's first relevant hit from rank 8 to 9. So section-title work
cannot meaningfully move these metrics by construction. Improving titles is a **provenance and
display** improvement, not a retrieval one, and should be judged on that basis — not on this
evaluation. The trade it offers is honest labels (56% coverage, few artifacts) against high
coverage carrying roughly 30% artifacts.

---

## EXPERIMENT 4 — Optional dense + cross-encoder reranker

**Baseline:** fresh baseline.
**Change:** new opt-in module. Dense retrieval unchanged and still the default.
**Files added:** `notebooks/clinical_rerank.py`. **No existing file modified.**

Pipeline: dense top-N candidates → `cross-encoder/ms-marco-MiniLM-L-6-v2` scores
`(query, chunk_text)` → top-10. No BM25, no RRF, no hybrid scoring, no query-specific rules,
keyword bonuses, per-document weights or recommendation-ID boosts. The reranker sees only the
query string and the chunk text.

| Metric | Baseline (dense) | + rerank, 30 cand. | Δ | + rerank, 20 cand. | Δ |
|---|---:|---:|---:|---:|---:|
| P@1 | 0.5000 | 0.5000 | 0 | 0.5000 | 0 |
| P@3 | 0.3667 | 0.3333 | −0.0334 | 0.3000 | −0.0667 |
| P@5 | 0.2800 | 0.2400 | −0.0400 | 0.2200 | −0.0600 |
| MRR | 0.5625 | 0.5736 | **+0.0111** | 0.5500 | −0.0125 |
| Recall@5 | 0.2217 | 0.1817 | −0.0400 | 0.2067 | −0.0150 |
| Recall@10 | 0.2667 | 0.2917 | +0.0250 | 0.2717 | +0.0050 |
| Relevant Top-1 | 5 | 5 | 0 | 5 | 0 |

Per query, 30 candidates — **4 improved, 3 regressed, 3 unchanged**:

| Q | First relevant: base → rerank | Note |
|---|---|---|
| 3 | 1 → 2 | lost a correct top-1 |
| 4 | — → 9 | previously unanswered, now found |
| 9 | 2 → 1 | gained a correct top-1 |
| 2, 7 | 1 → 1 | precision@5 fell (0.60→0.40, 0.80→0.40) |
| 6, 8 | 1 → 1 | recall improved |

**Decision: KEEP as an opt-in module; do NOT make it the default.**
**Reason:** at 30 candidates it trades precision for a marginal MRR gain — P@3, P@5 and
Recall@5 all fall, and it swaps one correct top-1 (Q3) for another (Q9), leaving P@1 and
Relevant Top-1 unmoved. At 20 candidates it is worse on every priority metric. That is not the
"meaningful overall improvement" required to change default behaviour. It is retained because
the brief asked for it to exist as an option; default retrieval is byte-identical to baseline
and does not import it. Delete `notebooks/clinical_rerank.py` for a strict revert.

Reproduce: `python eval/evaluate.py --label x --rerank --candidates 30`.

---

## Final state

| Category | Files |
|---|---|
| **Pipeline (unchanged, checksum-verified)** | `clinical_chunking.py`, `clinical_preprocess.py`, `clinical_rag.py`, all 3 notebooks, `tests/test_chunking.py`, all `data/processed/*`, all 4 PDFs, `README.md`, `requirements.txt` |
| **Added — evaluation** | `eval/gold_standard.json` (frozen), `eval/evaluate.py`, `eval/compare.py`, `eval/README.md`, `eval/runs/*.json` |
| **Added — documentation** | `docs/BASELINE_SNAPSHOT.md`, `docs/RETRIEVAL_OPTIMIZATION_EXPERIMENTS.md` |
| **Added — opt-in, not wired in** | `notebooks/clinical_rerank.py` |
| **Reverted** | Experiments 1, 2, 3 — no trace left in the repository |
| **Never written to the repo** | Experiment 5 — built and scored in an isolated directory; only its saved run `eval/runs/exp5_bge_base_en_v15.json` remains, as evidence |

Final verification: all 18 pre-existing source and data files match their recorded baseline
SHA-256; chunks 2,116; indexed 1,330; vectors (1330, 384) float32; alignment valid; max 254
tokens; 0 over limit; **28/28 tests pass**; final evaluation reproduces the fresh baseline
exactly (P@1 0.5000, MRR 0.5625).

## What the evidence says about where the ceiling actually is

Three of four experiments could not move P@1 at all, and the failures are concentrated in the
same three queries throughout: **Q4 (indications for EVAR), Q5 (risk factors), Q10 (open repair
vs EVAR)** never surface a relevant chunk in the top 10 under any configuration. Only the
cross-encoder ever reached one of them (Q4 at rank 9), and only by paying for it elsewhere.

That pattern points away from chunking and metadata. All three are *comparative or
multi-faceted* questions whose answers are spread across several recommendations in different
documents, and the retriever is a 384-dimensional bi-encoder with a 256-token window scoring a
single embedding per chunk. Reshaping chunk boundaries, deleting footers or fixing section
labels cannot fix a representation limit. The candidates most likely to move these numbers —
none of which were in scope here — are a stronger or domain-adapted embedding model, embedding
the section/recommendation context together with the chunk text, or query expansion. Each would
need the same frozen harness and the same standard of evidence.

**Experiment 5 tested the first of those and confirmed the diagnosis.** A stronger encoder moved
Q4, Q5 and Q10 — all three — for the first time, and lifted Recall@10 by 68% relative, while
four experiments' worth of chunking and metadata work had moved them exactly zero. The
bottleneck was the representation, as suspected. What Experiment 5 also shows is that a stronger
general-purpose encoder is not a free upgrade here: it buys breadth at the cost of the top-1
discriminations the baseline already won. The open question it leaves is whether breadth and
top-1 accuracy can be had together — which is where a domain-adapted (biomedical) encoder, or
BGE plus the already-built cross-encoder as a second stage, would be the next things to test.

**Experiment 6 tested the second of those and it failed.** A general-domain MS-MARCO reranker
does not repair BGE's ordering; it degrades it, promoting deep irrelevant candidates over
correct top hits. That closes off the cheap fix.

**Experiment 7 tested the first — a biomedical encoder — and it worked.** `MedEmbed-base-v0.1`
holds P@1 at the baseline's 0.5000 and Relevant Top-1 at 5/10 while lifting MRR +0.0569,
Recall@5 +0.135 and Recall@10 +0.2016. It is the only configuration in seven experiments to
improve retrieval without paying for it in top-1 accuracy, and it confirms the diagnosis
twice over: the ceiling was the representation, and the fix that works is *domain* adaptation
rather than raw encoder strength (BGE) or re-ordering (cross-encoder). The residual gap is Q4,
which no configuration except BGE-alone has ever surfaced.

---

## EXPERIMENT 5 — Embedding model swap (`all-MiniLM-L6-v2` → `BAAI/bge-base-en-v1.5`)

**Baseline:** fresh baseline, re-verified before starting (all 18 source/data files matched their
recorded SHA-256; gold hash `0b8a443b…` unchanged; metrics reproduced exactly).
**Change:** the embedding model, and nothing else.
**Files modified in the repository:** none. The experimental index was built in an isolated
directory outside the repo; `data/embeddings/` was never written to.

Held constant: the 4 PDFs, extracted text, all 2,116 chunks and their boundaries, the 1,330
indexed chunks and their order, all metadata, the clinical filtering, the 10 queries, the frozen
gold standard, the metric implementation, and the dense cosine retrieval path
(`clinical_rag.retrieve`, reused unmodified). No reranking, BM25, RRF, query expansion or
instruction prefix — BGE's optional query instruction was deliberately **not** used, so queries
and passages are encoded exactly as the baseline encodes them.

### Why this model

Chosen and committed **before** any experimental result was seen. `bge-base-en-v1.5` scores
~53 nDCG@10 on the MTEB retrieval average against ~42 for `all-MiniLM-L6-v2` — a large gap on
dense passage retrieval specifically, which is the bottleneck under test. It is 109M parameters
(~0.44 GB), runs on CPU, needs no API key, and is a standard sentence-transformers model whose
intended metric is cosine over L2-normalised vectors — a drop-in for this architecture. Its
512-token window comfortably clears the existing 254-token chunks, so **chunk size was not
changed**. `all-mpnet-base-v2` was rejected as too small an improvement (~44) to test the
hypothesis; the `e5` family was rejected because its mandatory `query:`/`passage:` prefixes
would have been a prompt change. Only this one model was run.

### Index validation

| Check | Result |
|---|---|
| Chunks embedded | 1,330 (= baseline indexed count) |
| chunk_id uniqueness / all present in `chunks.json` | ✅ / ✅ |
| `chunk_text` identical to baseline, per chunk | ✅ |
| Max chunk tokens under BGE's own tokenizer | 254 |
| Chunks over the model window (512) | **0** |
| Vector shape / dtype | (1330, 768) float32 |
| Vectors finite | ✅ |
| L2 norms | 0.99999988 – 1.00000012 (normalised) |
| All-zero vectors | none |
| Baseline artifacts after the build | `chunks.json`, `embeddings.npy`, `embedded_chunks.json` all byte-identical |

### Result

| Metric | Baseline (MiniLM, 384d) | BGE-base (768d) | Δ |
|---|---:|---:|---:|
| **P@1** | **0.5000** | **0.4000** | **−0.1000** |
| P@3 | 0.3667 | 0.4000 | +0.0333 |
| P@5 | 0.2800 | 0.4000 | +0.1200 |
| **MRR** | 0.5625 | 0.6167 | +0.0542 |
| Recall@5 | 0.2217 | 0.3250 | +0.1033 |
| Recall@10 | 0.2667 | 0.4483 | +0.1816 |
| **Relevant Top-1** | **5/10** | **4/10** | **−1** |
| Answering@5 | 6/10 | **10/10** | +4 |

**Improved: 6 (Q1, Q4, Q5, Q6, Q8, Q10). Regressed: 4 (Q2, Q3, Q7, Q9). Unchanged: 0.**
Tests: 28/28 passing.

The three queries that no previous configuration could answer all improved:

| Q | Baseline first relevant | BGE first relevant |
|---|---|---|
| Q4 indications for EVAR | none in top-10 | **rank 3** |
| Q5 risk factors for AAA | none in top-10 | **rank 1** |
| Q10 open repair vs EVAR | none in top-10 | **rank 4** |

But top-1 accuracy fell on queries the baseline already got right:

| Q | Baseline first relevant | BGE first relevant |
|---|---|---|
| Q2 repair diameter | 1 | 2 |
| Q3 surveillance | 1 | 3 |
| Q9 women and screening | 2 | 4 |

**Decision: DO NOT MAKE DEFAULT — retained as a validated optional configuration.**
**Reason:** P@1 is the top-priority metric and it fell 0.50 → 0.40, with Relevant Top-1 dropping
5 → 4. The pre-registered rule for exactly this outcome — "if the model improves deeper recall
but damages top-1 clinical accuracy, report it as an optional experiment, NOT the default" —
applies directly. Nothing in the repository was changed, so no revert was required.

This is nonetheless the most informative result in the whole series, and the first evidence that
the earlier ceiling hypothesis was right: Q4/Q5/Q10 were never a chunking, footer or
section-title problem, they were a representation problem, and a stronger encoder moved all
three. Recall@10 rose 68% relative, and every one of the 10 queries now surfaces a relevant
chunk within the top 5. The cost is that the finer-grained top-1 discriminations the baseline
happened to win are lost.

Reproduce: build an index over `data/embeddings/embedded_chunks.json` with
`BAAI/bge-base-en-v1.5` (`normalize_embeddings=True`, no instruction prefix) and score it with
`eval/evaluate.py`'s frozen `evaluate_run`. Saved run: `eval/runs/exp5_bge_base_en_v15.json`.

---

## EXPERIMENT 6 — BGE-base top-30 → cross-encoder → top-10

**Hypothesis:** BGE improves candidate *recall* but sometimes orders the wrong passage first;
the Experiment 4 cross-encoder should fix the ordering and give better P@1/MRR while keeping
BGE's recall gains.

**Configuration** — one variable, reranking applied on top of BGE candidates:

| Stage | Setting |
|---|---|
| Embedding model | `BAAI/bge-base-en-v1.5` (768d, isolated index, 1,330 vectors) |
| Candidate depth | **30** (set explicitly) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` — the exact model from Experiment 4, unchanged |
| Evaluation depth | **10** |

No BM25, no RRF, no hybrid, no query expansion. Chunk size, overlap, preprocessing, section
detection and the gold standard were all untouched. Query and passage text were passed to the
reranker verbatim. **Files modified in the repository: none.**

### Result — the hypothesis is refuted

| Metric | MiniLM baseline | BGE only (Exp 5) | **BGE + cross-encoder** | Δ vs baseline | Δ vs BGE only |
|---|---:|---:|---:|---:|---:|
| **P@1** | **0.5000** | 0.4000 | **0.4000** | **−0.1000** | 0 |
| P@3 | 0.3667 | 0.4000 | 0.3333 | −0.0334 | −0.0667 |
| P@5 | 0.2800 | 0.4000 | 0.3000 | +0.0200 | −0.1000 |
| **MRR** | **0.5625** | 0.6167 | **0.5294** | **−0.0331** | **−0.0873** |
| Recall@5 | 0.2217 | 0.3250 | 0.2767 | +0.0550 | −0.0483 |
| Recall@10 | 0.2667 | 0.4483 | 0.3750 | +0.1083 | −0.0733 |
| Relevant Top-1 | 5/10 | 4/10 | 4/10 | −1 | 0 |
| Answering@5 | 6/10 | 10/10 | 7/10 | +1 | **−3** |

Reranking did not repair BGE's ordering — it **degraded BGE on six of eight metrics** and left
the other two tied. Against the production baseline, both priority metrics fall: P@1 −0.10 and
MRR −0.0331.

### Where the reranker went wrong

Tracing each final top-1 back to the BGE rank it came from shows a consistent failure: the
cross-encoder repeatedly promotes deep, irrelevant candidates over BGE's correct top hits.

| Q | BGE first relevant (in top-30) | Final first relevant | Final top-1 came from | Relevant? |
|---|---:|---:|---|---|
| 1 | 2 | 10 | BGE #3 | ✗ |
| 2 | 2 | **1** | BGE #2 | ✓ |
| 3 | 3 | 4 | BGE #2 | ✗ |
| 4 | 3 | **none in top-10** | BGE #11 | ✗ |
| 5 | **1** | 3 | BGE #11 | ✗ |
| 6 | 1 | 1 | BGE #6 | ✓ |
| 7 | 1 | 1 | BGE #1 | ✓ |
| 8 | 1 | 1 | BGE #1 | ✓ |
| 9 | 4 | 2 | BGE #12 | ✗ |
| 10 | 4 | 9 | BGE #5 | ✗ |

The clearest refutations: **Q4** had a relevant chunk at BGE rank 3 and the reranker pushed it
out of the top 10 entirely, making the query unanswerable again; **Q5** had a relevant chunk at
BGE rank **1** and the reranker demoted it to 3 in favour of BGE #11. In five of ten queries the
promoted rank-1 was a candidate BGE had ranked 5th or deeper, and in four of those it was not
relevant.

This is a domain-mismatch result, not a tuning problem. `ms-marco-MiniLM-L-6-v2` is trained on
short general-web query/passage pairs; asked to order dense clinical guideline prose where many
candidates share heavy terminology overlap, its scores are less reliable than BGE's own
similarity. The same reranker showed mixed query-level effects on MiniLM candidates in
Experiment 4; on stronger BGE candidates it is actively harmful, because there is now a better
ordering for it to destroy.

**Decision: DO NOT MAKE DEFAULT.**
**Reason:** the pre-registered rule — "if recall improves but P@1/MRR become materially worse,
DO NOT MAKE DEFAULT" — is met exactly. Against the production baseline P@1 falls 0.10 and MRR
falls 0.033; against BGE alone it is worse on six of eight metrics. Nothing was changed in the
repository, so no revert was required.

Saved run: `eval/runs/exp6_bge_plus_crossencoder.json` (includes the full BGE-rank →
cross-encoder-rank trace for every query).

---

## EXPERIMENT 7 — Biomedical encoder (`abhinand/MedEmbed-base-v0.1`)

**Hypothesis:** a biomedical/clinical-domain encoder may preserve or improve top-1 accuracy
while retaining BGE's recall advantage.

**Model:** `abhinand/MedEmbed-base-v0.1`, revision `7a90c50263f620dff743eb9794b89a42bfc5d765`,
768d, 512-token window. Selected and frozen **before** any result was seen; only this one model
was evaluated.

Chosen for experimental-design reasons as much as domain: MedEmbed-base is a **medical-retrieval
fine-tune of the BGE-base architecture**, so comparing it against Experiment 5 holds backbone
family, dimension, pooling and normalisation constant and isolates domain adaptation as the sole
variable. A PubMedBERT-based alternative (`NeuML/pubmedbert-base-embeddings`) would have
confounded domain with architecture; `pritamdeka/S-PubMedBert-MS-MARCO` was rejected because
Experiment 6 already showed MS-MARCO-trained models transfer badly to this corpus. Used without
instruction prefixes, so query wording and encoding procedure are identical to baseline.

**Files modified in the repository: none.** Index built in an isolated directory outside the
repo; `data/embeddings/` never written to. Chunks, chunk IDs, chunk text, metadata, filtering,
queries, evaluator, gold standard and evaluation depth all held constant.

### Index validation

| Check | Result |
|---|---|
| Chunks embedded | 1,330 (identical set, no additions/removals) |
| chunk_id unique / all present in `chunks.json` | ✅ / ✅ |
| `chunk_text` byte-identical to baseline | ✅ 1,330/1,330 |
| Max chunk tokens (MedEmbed tokenizer) | 254 of a 512 window |
| Hidden truncation | **none** — 0 chunks over the window |
| Vector shape / dtype | (1330, 768) float32 |
| Finite / zero vectors | all finite / 0 zero vectors |
| L2 norms | 0.99999988 – 1.00000012 |
| Query encoding | same model, same procedure |

### Result — the first experiment to clear the bar

| Metric | MiniLM baseline | BGE (Exp 5) | **MedEmbed (Exp 7)** | Δ vs baseline | Δ vs BGE |
|---|---:|---:|---:|---:|---:|
| **P@1** | **0.5000** | 0.4000 | **0.5000** | **0** | **+0.1000** |
| P@3 | 0.3667 | 0.4000 | 0.3333 | −0.0334 | −0.0667 |
| P@5 | 0.2800 | 0.4000 | 0.4000 | +0.1200 | 0 |
| **MRR** | 0.5625 | 0.6167 | **0.6194** | **+0.0569** | +0.0027 |
| Recall@5 | 0.2217 | 0.3250 | **0.3567** | **+0.1350** | +0.0317 |
| Recall@10 | 0.2667 | 0.4483 | **0.4683** | **+0.2016** | +0.0200 |
| **Relevant Top-1** | **5/10** | 4/10 | **5/10** | **0** | +1 |
| Answering@5 | 6/10 | 10/10 | 8/10 | +2 | −2 |

vs baseline: **5 improved, 4 regressed, 1 unchanged.** Tests 28/28.

MedEmbed recovers the P@1 that BGE lost while keeping — and slightly exceeding — BGE's recall.
It is better than BGE on six of eight metrics, and better than the baseline on every metric
except P@3.

### Top-1 composition changed, count did not

| | Queries correct at rank 1 |
|---|---|
| Baseline | Q2, Q3, Q6, Q7, Q8 |
| MedEmbed | Q1, Q2, Q5, Q7, Q8 |

Gained Q1 (screening, was rank 8) and Q5 (risk factors, was unretrievable). Lost Q3
(surveillance → rank 2) and Q6 (imaging → rank 3). The two losses stay inside the top 3 and both
gain deeper recall (Q3 Recall@10 0.33 → 0.67; Q6 Recall@5 0.25 → 0.50), so this is a reshuffle
within the top ranks rather than a systematic displacement of clinical answers.

### The previously unreachable queries

| Q | Baseline | BGE | MedEmbed |
|---|---|---|---|
| Q4 indications for EVAR | none in top-10 | rank 3 | **still none in top-10** |
| Q5 risk factors | none in top-10 | rank 1 | **rank 1** |
| Q10 open repair vs EVAR | none in top-10 | rank 4 | rank 9 |

Two of three fixed and held. **Q4 remains unsolved by every configuration tested except BGE
alone** — it is the single hardest query in the set and the honest gap in this result.

**Decision: MEETS THE KEEP CRITERIA — not applied.**
Against the pre-registered rule: P@1 ≥ 0.5000 ✅ (exactly 0.5000); MRR ≥ 0.5625 ✅ (0.6194);
Recall@10 does not materially regress ✅ (+0.2016); no systematic baseline top-1 regression ✅
(5/10 preserved, losses land at ranks 2–3). It also satisfies the STRONG CANDIDATE clause except
that Q4 did not become more retrievable.

Saved run: `eval/runs/exp7_medembed_base_v01.json`.

### Adoption into production

Subsequently authorised and applied as a single controlled change. `DEFAULT_EMBED_MODEL` moved
to `abhinand/MedEmbed-base-v0.1` with the revision **pinned by commit**, `MODEL_MAX_TOKENS`
raised 256 → 512, and a single `load_embedder()` entry point introduced so chunking, index
building and query encoding can never drift onto different weights. Chunks were **not** rebuilt:
`data/chunks/chunks.json` is byte-identical (`a07bc49a…`), as is `embedded_chunks.json`
(`4a27dace…`) — only `embeddings.npy` and `index_meta.json` changed.

The production index reproduces the isolated experiment **exactly**: all eight metrics identical
and all ten per-query first-relevant ranks identical (0 improved / 0 regressed / 10 unchanged).

Tests went 28 → 29. No test was weakened: the two model-specific assertions now check against
the *active* model rather than the literal 256 (`model_token_limit()` must equal the loaded
model's `max_seq_length`; the oversize-refusal test derives its threshold from the active
limit), and one test was **added** asserting the production model is revision-pinned.

Saved production run: `eval/runs/phase1_production_medembed.json`.

---

## Appendix — Experiment 2 implementation (reverted, kept for reference)

Not in the repository. To reapply, add this to `notebooks/clinical_chunking.py` above
`is_guideline_document`, and replace the page loop header in `build_chunks` with the two lines
shown at the end.

```python
REPEAT_MIN_PAGES = 3
REPEAT_MIN_FRACTION = 0.25
EDGE_HEAD_LINES = 2
EDGE_TAIL_LINES = 3
EDGE_FRACTION = 0.90
MAX_FURNITURE_LINE_CHARS = 200

_DIGIT_RUN_RE = re.compile(r"\d+")
_MEASUREMENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:cm|mm|%|mg|years?|months?|weeks?)", re.I)
_NUMBERED_ITEM_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_HEADING_WORD_RE = re.compile(r"\brecommendations?\b", re.I)
_FURNITURE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"page\s*\d+\s*(?:of\s*\d*)?"
    r"|\d{1,4}"
    r"|(?:\d{0,4}\s*)?[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]*){0,3}\s+et\s+al\.?\s*\d*"
    r"|.{0,60}all rights reserved.*"
    r"|.{0,60}subject to notice of rights.*"
    r"|(?:©|\(c\))\s*\d{0,4}.*"
    r"|https?://\S+|www\.\S+"
    r"|\(reprinted\)(?:\s*jama\.com)?"
    r"|jama\.com|jama"
    r"|volume\s*\d+\s*,\s*number\s*\d+"
    r"|.*clinical review & education.*"
    r")\s*$",
    re.I,
)


def _line_shape(line: str) -> str:
    """Digit-insensitive form, so 'Page 17 of 53' and 'Page 18 of 53' collapse."""
    return _DIGIT_RUN_RE.sub("#", re.sub(r"\s+", " ", line).strip().lower())


def line_carries_clinical_signal(line: str) -> bool:
    """Guard: a line that could be clinical content is never treated as furniture."""
    return bool(
        _RECOMMENDATION_RE.search(line)
        or _CLINICAL_SIGNAL_RE.search(line)
        or _HEADING_WORD_RE.search(line)
        or _MEASUREMENT_RE.search(line)
        or _NUMBERED_ITEM_RE.search(line)
    )


def detect_repeated_furniture(page_texts: list[str]) -> set[str]:
    """Line shapes that are page furniture across this document."""
    n_pages = len([t for t in page_texts if (t or "").strip()])
    if n_pages < REPEAT_MIN_PAGES:
        return set()
    threshold = max(REPEAT_MIN_PAGES, int(round(REPEAT_MIN_FRACTION * n_pages)))

    pages_seen: Counter[str] = Counter()
    edge_hits: Counter[str] = Counter()
    total_hits: Counter[str] = Counter()
    sample: dict[str, str] = {}
    for text in page_texts:
        lines = [ln for ln in (text or "").split("\n")]
        n = len(lines)
        on_this_page: set[str] = set()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            shape = _line_shape(line)
            on_this_page.add(shape)
            total_hits[shape] += 1
            if i < EDGE_HEAD_LINES or i >= n - EDGE_TAIL_LINES:
                edge_hits[shape] += 1
            sample.setdefault(shape, line.strip())
        for shape in on_this_page:
            pages_seen[shape] += 1

    furniture = set()
    for shape, seen in pages_seen.items():
        if seen < threshold or not 2 <= len(shape) <= MAX_FURNITURE_LINE_CHARS:
            continue
        line = sample[shape]
        if line_carries_clinical_signal(line):
            continue
        edge_anchored = edge_hits[shape] / total_hits[shape] >= EDGE_FRACTION
        if edge_anchored or _FURNITURE_LINE_RE.match(line):
            furniture.add(shape)
    return furniture


def strip_repeated_furniture(text: str, furniture: set[str]) -> str:
    if not furniture or not text:
        return text
    kept = [ln for ln in text.split("\n") if not (ln.strip() and _line_shape(ln) in furniture)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
```

In `build_chunks`, the per-document page loop header becomes:

```python
        rows = doc_pages.to_dict("records")
        page_texts = [page_working_text(r) for r in rows]
        furniture = detect_repeated_furniture(page_texts)
        for row, raw_text in zip(rows, page_texts):
            text = strip_repeated_furniture(raw_text, furniture)
            status = row.get("extraction_status")
            ...  # loop body unchanged
```
