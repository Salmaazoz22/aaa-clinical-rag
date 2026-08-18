# AAA Clinical RAG

> ⚠️ **This is a clinical information retrieval / RAG research prototype. It has NOT been
> clinically validated as a decision-support system.** It retrieves guideline passages; it does
> not generate advice, and no claim of clinical safety, accuracy or fitness for patient care is
> made. The corpus contains genuinely **conflicting** recommendations across guidelines, which
> the system surfaces without reconciling. Not for clinical use.

Retrieval over four abdominal aortic aneurysm (AAA) clinical guidelines, with an evidence-first
evaluation designed so that its own results can be checked and, where necessary, disbelieved.

**Retrieval is dense cosine similarity only.** No query rewriting, no intent detection, no
keyword bonuses, no per-question rules. Nothing in the retrieval path can branch on *which*
question is being asked — a rule that cannot see the question cannot be fitted to it.

---

## Results

Three frozen question sets, **reported separately and never pooled**. `final20` is the only one
that was authored and hash-frozen *before* the configuration being tested existed.

| set | config | P@1 | P@3 | P@5 | MRR | R@5 | R@10 | Rel@1 | Ans@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original10 | baseline (page-buffer, historical) | 0.500 | 0.333 | 0.400 | 0.619 | 0.357 | 0.468 | 5/10 | 8/10 |
| original10 | **V1 atomic (SHIPPED)** | 0.600 | 0.433 | 0.400 | 0.775 | 0.415 | 0.558 | 6/10 | 10/10 |
| heldout18 | baseline (page-buffer, historical) | 0.556 | 0.370 | 0.344 | 0.697 | 0.667 | 0.824 | 10/18 | 16/18 |
| heldout18 | V1 atomic — as first published (superseded) | 0.722 | 0.556 | 0.444 | 0.815 | 0.769 | 0.898 | 13/18 | 17/18 |
| heldout18 | **V1 atomic (SHIPPED, recomputed)** | 0.722 | **0.574** | **0.456** | **0.824** | 0.769 | 0.898 | 13/18 | 17/18 |
| **final20** | baseline (page-buffer, historical) | 0.400 | 0.317 | 0.250 | 0.531 | 0.542 | 0.608 | 8/20 | 14/20 |
| **final20** | **V1 atomic (SHIPPED)** | **0.550** | 0.333 | 0.300 | **0.664** | **0.667** | **0.783** | **11/20** | 16/20 |

The retriever is **identical** in every row — same model, same pinned revision, same dense
cosine, no reranking. The only variable is where chunk boundaries fall.

#### Correction: the heldout18 V1 row was recomputed on the shipped chunker

The V1 rows were originally produced by `eval/experimental_atomic_chunking.py`, which carried a
second copy of the atomic chunker. That copy emitted **1,764 chunks / 1,004 indexed** where the
shipped chunker emits **1,760 / 991** — the difference being the citation-heading fix, which was
off when those rows were written. `original10` and `final20` are unaffected (every metric
identical), but three `heldout18` metrics moved, all upward:

| metric | as first published | shipped chunker | Δ |
|---|---:|---:|---:|
| P@3 | 0.5556 | **0.5741** | +0.0185 |
| P@5 | 0.4444 | **0.4556** | +0.0112 |
| MRR | 0.8148 | **0.8241** | +0.0093 |

The entire delta comes from one question — heldout18 Q7, "What cardiac assessment is needed before
aneurysm repair?" — whose first relevant hit moved from rank 3 to rank 2. No other question changed
any scored quantity. Seven other questions retrieved differently-numbered chunk ids from the same
pages, which the page-range relevance rule scores identically.

The superseded row is retained above rather than deleted. Artifact:
`eval/runs/p1_shipped_chunker_all_sets.json`, which records both sets of numbers, the per-question
detail, and a control confirming `final20` still reproduces
`eval/runs/final_corrected_v1_final20.json` exactly. The two chunker implementations have since
been collapsed into one (`notebooks/clinical_atomic_chunking.py`); see
`docs/REFERENCE_COMPARISON.md`.

**Decision: ADOPT WITH CAVEATS** (`eval/final_recommendation.md`). V1 is the first change in the
project's history to raise P@1, and the only one to do so on a pre-registered set. **It is now the
shipped default.**

### Final corrected validation (the promotion gate)

Every V1/V2 number above **except the recomputed heldout18 row** was produced with a known anchor
defect present — numbered *bibliography*
lines were being accepted as section headings (all 15 USPSTF "sections" were reference entries).
The gate turned the fix on and re-ran V1 against `final20` only, tuning nothing:

| metric | historical V1 (fix off) | corrected V1 (fix ON, shipped) | Δ |
|---|---:|---:|---:|
| P@1 | 0.5500 | 0.5500 | **+0.0000** |
| MRR | 0.6642 | 0.6642 | **+0.0000** |
| Recall@10 | 0.7833 | 0.7833 | **+0.0000** |

All eight metrics identical. 15 bogus USPSTF and 13 bogus ESVS section anchors removed;
recommendation anchors untouched. Artifact: `eval/runs/final_corrected_v1_final20.json`.

### Why the highest-scoring configuration was rejected

`V2_atomic_pure` posts the best raw numbers in the project (original10 P@1 0.700, MRR 0.820) and
was **rejected**. The frozen relevance rule requires the chunk's page range to overlap the answer
passage's page range, and V2's chunks average 5.57 pages. A control (`V4`) took the production
index, changed **nothing** about retrieval, and merely widened page metadata — that alone was
worth **P@1 +0.10**. Re-scoring every chunk on its start page only removes the term entirely:

| config | orig-10 P@1 scored → strict | held-18 | final20 |
|---|---|---|---|
| baseline | 0.500 → 0.500 | 0.556 → 0.556 | 0.400 → 0.400 |
| **V1** | 0.600 → **0.600** | 0.722 → **0.722** | 0.550 → **0.550** |
| V2 | 0.700 → **0.400** | 0.722 → 0.667 | 0.550 → 0.550 |

V1's gain is unchanged with and without the confound, on all three sets. V2's is not.

---

## Corpus and index

| | |
|---|---|
| Guidelines | USPSTF 2019 · NICE NG156 · ESVS 2024 · SVS 2018 |
| Pages | 249 |
| Chunker | **atomic / structure-driven (V1)** — `notebooks/clinical_atomic_chunking.py` |
| Chunks | 1,760 total → **991 indexed** (references, contents pages, boilerplate and title-only slides are labelled and excluded, never deleted) |
| Embedding | `abhinand/MedEmbed-base-v0.1`, revision `7a90c50263f620dff743eb9794b89a42bfc5d765` |
| Vectors | 991 × 768, float32, L2-normalised |
| Token safety | max 512 tokens, **0** over the model window; the validator raises rather than truncating |
| Index | `numpy_cosine`, exhaustive |
| Latency | ~32 ms/query, CPU only |

**All 48 questions across the three sets are answerable from the index.** Poor retrieval is a
retrieval failure, not a coverage gap, so adding documents is **not** recommended
(`eval/corpus_audit.json`).

---

## Layout

```
data/                     PDFs, extracted pages, chunks, embeddings (the shipped V1 index)
  archive_baseline_index/   the previous page-buffer index, preserved intact
notebooks/                pipeline modules + notebooks
  clinical_preprocess.py    PDF -> pages, sections, recommendations
  clinical_chunking.py      token budget, validation, chunker selection
  clinical_atomic_chunking.py  ** the shipped chunker (V1, structure-driven) **
  clinical_rag.py           embeddings, index, retrieval
  clinical_rerank.py        optional cross-encoder (NOT wired into production)
  final_evaluation.ipynb    ** the presentation notebook, 23 sections **
eval/                     gold standards, evaluator, experiments, evidence
  gold_standard.json          10 questions   (frozen)
  gold_standard_heldout.json  18 questions   (frozen)
  gold_standard_final20.json  20 questions   (frozen; SHA in .sha256)
  evaluate.py                 the frozen, retriever-agnostic evaluator
  final_evidence.json         per-query retrieved evidence (pre-promotion index)
  runs/final_corrected_v1_final20.json  evidence for the SHIPPED config
  final_evaluation_results.json
  experiment_history.json     23 experiments
  runs/                       every historical run, preserved
docs/                     experiment_history.md · presentation_story.md
                          PROJECT_B_LESSONS.md · limitations.md · deployment_readiness.md
tests/                    29 tests
aaa-clinical-ragnour/     Project B - EXTERNAL read-only reference. Not a dependency,
                          not published to git.
aaa-clinical-raggehad/    A separate person's project variant. Untouched, not published to git.
```

## Reproduce

```bash
pip install -r requirements.txt

python eval/evaluate.py --label baseline        # shipped config, original 10
python eval/run_final_evaluation.py             # baseline vs V1 vs V2, all 3 sets (~45 min CPU)
python eval/rescore_conservative.py             # removes the page-overlap confound (instant)
python eval/audit_corpus_and_questions.py       # corpus + question audits
python eval/run_stability_checks.py             # 16 stability / readiness checks
python eval/verify_integrity.py                 # 19 integrity checks
python eval/rebuild_shipped_index.py            # rebuild data/chunks + data/embeddings
python eval/build_experiment_history.py         # regenerates history JSON + markdown
python -m pytest tests -q                       # 29 tests
```

**Open the presentation notebook** (already executed; outputs are committed):

```bash
jupyter notebook notebooks/final_evaluation.ipynb
# or re-execute end to end:
jupyter nbconvert --to notebook --execute --inplace notebooks/final_evaluation.ipynb
```

**Where the evidence lives:** frozen question sets and the evaluator in `eval/`; per-query
retrieved evidence in `eval/final_evidence.json` (historical index) and
`eval/runs/final_corrected_v1_final20.json` (shipped config); every experiment ever run in
`eval/runs/`; the full narrative in `docs/experiment_history.md`; SHA-256 of every frozen
artifact in `eval/final_artifact_hashes.json`.

> **Reproduction boundary.** The shipped artifacts reproduce from the shipped code
> (`python eval/rebuild_shipped_index.py`). Every run in `eval/runs/` and the tables in
> `eval/final_evaluation_results.json` were scored against the **previous** page-buffer index,
> preserved in `data/archive_baseline_index/`; they were not recomputed. That chunker is still
> reachable via `run_chunking(strategy="page_buffer")`.

## Status

**Not production ready, and not claimed to be.** The retrieval core is deterministic,
reproducible, token-safe and robust to malformed input (14/16 stability checks pass, 19/19
integrity checks pass). There is no service,
no logging, no authentication, and **no abstention threshold** — an out-of-scope query still
returns 10 chunks. See `docs/deployment_readiness.md`.

## Project B

`aaa-clinical-ragnour/` (renamed from `aaa-clinical-rag/`) is a **separate** project kept as read-only comparison evidence. Nothing in
it was modified or re-run, and **the final project does not depend on it in any way**.

Its chunking idea was audited, isolated and tested — that is where V1 comes from. Its retrieval
machinery was refused: 10 intents for 10 evaluation questions, boosts naming specific answers by
ID and literal sentence, and a checker whose "correct answers" are **9 of 10** literal strings
copied from its own ranker's scoring rules. Measured by running its code: intents fire 10/10 on
its own questions and **1/18** on ours. It also silently truncates **143 of 452** indexed chunks.
Full audit: `eval/project_b_comparison.json`, `docs/PROJECT_B_LESSONS.md`.
