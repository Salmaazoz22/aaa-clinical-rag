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
been collapsed into one (`ingestion/atomic_chunking.py`); see
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
| Chunker | **atomic / structure-driven (V1)** — `ingestion/atomic_chunking.py` |
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

## Production Vector Database

Qdrant is the **production storage and retrieval backend**. The local numpy index is
**preserved unchanged** as the reproducibility artifact — every published number above was
produced from it, and it remains the reference the production store is verified against.

| | research artifact | production backend |
|---|---|---|
| where | `data/embeddings/` (numpy, exhaustive cosine) | Qdrant collection `aaa_clinical_v1` |
| role | what the frozen evaluation was run on | what a service queries |
| used by | `eval/`, `notebooks/` | `vectordb/retriever.py` (FastAPI later) |

**Same index, moved — not rebuilt.** Vector dimension **768**, distance **cosine**, top-K
**10**, **991** points, `abhinand/MedEmbed-base-v0.1` @ `7a90c502…`, L2-normalised. The
existing `embeddings.npy` values are uploaded as-is; nothing is re-chunked or re-embedded.
Point IDs are deterministic — `uuid5(fixed-namespace, chunk_id)` — and the original `chunk_id`
is preserved in the payload, which carries the complete indexed record (text, document,
document_id, page_number/start/end, section, content_type, token_count, char_count,
source_file, source_excerpt, recommendation_id/grade, evidence_level).

Retrieval semantics are untouched: question → MedEmbed embedding → cosine → top 10. No
reranking, no query rewriting, no intent detection, no filtering, no per-question rules, **no
LLM yet**.

### Local vs Qdrant equivalence

The migration is only accepted if the database returns what the numpy index returns. All 48
frozen questions (original10 + heldout18 + final20) are embedded once and sent down both paths:

| queries | same top-1 | same top-10 | same order | max abs. score diff | verdict |
|---:|---:|---:|---:|---:|---|
| 48 / 48 pass | 48/48 | 48/48 | 48/48 | 2.075e-07 (tol 1e-05) | **EQUIVALENT** |

Those question sets are used here **only** as a fixed query sample — no metric is computed, no
gold standard is scored, nothing is tuned. Artifact:
`eval/qdrant_migration_verification.json`. Full detail: `docs/vector_database.md`.

### Running it

```bash
pip install -r requirements.txt
cp .env.example .env                     # never commit .env

docker compose up -d                     # local Qdrant v1.19.0 :6333 / :6334
python vectordb/ingest.py --recreate     # migrate 991 vectors  (measured: 0.95 s)
python vectordb/verify_migration.py      # local vs Qdrant equivalence, 48 queries
python vectordb/benchmark.py             # latency + footprint
python vectordb/retriever.py "When is elective AAA repair recommended?"
```

Ingestion refuses to run on bad data rather than repairing it: wrong dimension, NaN/Inf or
unnormalised vectors, count mismatch, missing/duplicate chunk IDs, missing or null payload
fields, token-limit violations, or an `index_meta.json` that does not name the pinned model.

Environment (no credential is ever hardcoded or committed; see `.env.example`):
`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `QDRANT_PREFER_GRPC`, `QDRANT_TIMEOUT`,
`QDRANT_EXACT_SEARCH`, `QDRANT_LOCAL_PATH`. Qdrant Cloud is supported by pointing `QDRANT_URL`
at the cluster and setting `QDRANT_API_KEY`; the same equivalence check is the acceptance test
for any deployment target.

Measured on local Docker, CPU, 48 queries × 3: Qdrant search **9.8 ms mean / 26.3 ms p95**,
query embedding 39.5 ms, end-to-end 51.1 ms; collection 10.0 MiB, container ~92 MiB.

---

## API and demo UI

Two layers sit on top of the pipeline, and neither contains any of it.

```
Browser → Streamlit (ui/) → HTTP → FastAPI (api/) → generation.pipeline.answer_question
                                                     → safety gate
                                                     → MedEmbed → Qdrant
                                                     → evidence threshold
                                                     → LLM
                                                     → citation validator
```

`api/` is a transport layer: it calls `answer_question` as-is and serialises the
`GenerationResult` it gets back. `ui/` is an HTTP client: nothing under it imports
`generation`, `retrieval`, `vectordb` or `ingestion`, and `tests/test_ui.py` asserts that by
parsing every UI source. There is exactly one retrieval implementation in this project.

### Endpoints

| endpoint | returns |
| --- | --- |
| `GET /health` | API, Qdrant, index and LLM-key status |
| `GET /v1/meta` | pinned model + revision, collection, dimensions, chunk count, index digests |
| `POST /v1/answer` | the full audit record for one question, refusals included |
| `GET /v1/chunks/{chunk_id}` | one chunk's stored payload, for citation audit |
| `GET /v1/corpus` | the four guideline documents, with live per-document chunk counts |
| `GET /v1/evaluation` | the frozen evaluation artifacts, verbatim, with the SHA-256 of the bytes read |

`/v1/corpus` and `/v1/evaluation` exist so the UI can display corpus and metric values without
hardcoding them. They read frozen artifacts off disk and return them unchanged — they have no
way to produce any other number.

An upstream failure is never dressed up as an answer: a missing API key is `503`, an
unparseable or failed model call is `502`, and an invalid question is `400`. No placeholder,
cached or ungrounded answer is ever substituted.

### Running it

```bash
pip install -r requirements.txt
cp .env.example .env                     # never commit .env

docker compose up -d                     # Qdrant :6333  (or set QDRANT_LOCAL_PATH for embedded mode)
python vectordb/ingest.py --recreate     # migrate the 991 frozen vectors

uvicorn api.main:app --host 127.0.0.1 --port 8000    # terminal 1 — OpenAPI docs at /docs
streamlit run ui/app.py                              # terminal 2 — http://localhost:8501
```

The UI reads `CLINICAL_RAG_API_URL` to reach a backend elsewhere; it defaults to
`http://127.0.0.1:8000`. With the API down it shows a "backend unavailable" page explaining how
to start it, never a stack trace.

Retrieval, the safety gate and the evidence threshold need no LLM key. Only a question that
passes every gate reaches the model, so the refusal behaviour is fully demonstrable without one.

### UI pages

| page | what it shows |
| --- | --- |
| **Ask** | answer, confidence, citation count and validator verdict; retrieved evidence with per-chunk similarity and full text; the evidence → answer trace; retrieval-score bars against the evidence floor; validator findings; run provenance; the raw API response |
| **Evaluation** | the frozen retrieval metrics for all three question sets, every configuration, labelled frozen and served from `/v1/evaluation` with artifact digests |
| **Safety & Abstention** | five example questions *executed live*, showing which gate fired and why |
| **Architecture** | the request path as implemented, drawn from live configuration values |
| **Guidelines & Sources** | the four documents, from extraction metadata, with live chunk counts |
| **Technical Details** | each component, and why it is there rather than the obvious alternative |

Clinical answers are never cached. Static metadata is, keyed on the backend address so
repointing the UI cannot serve the previous backend's provenance.

---

## Layout

```
data/                     PDFs, extracted pages, chunks, embeddings (the shipped V1 index)
  archive_baseline_index/   the previous page-buffer index, preserved intact
ingestion/                ** corpus -> chunks **
  preprocess.py             PDF -> pages, sections, recommendations
  chunking.py               token budget, validation, chunker selection
  atomic_chunking.py        ** the shipped chunker (V1, structure-driven) **
retrieval/                ** chunks -> ranked evidence **
  index.py                  embeddings, local index, retrieval
  rerank.py                 optional cross-encoder (NOT wired into production)
api/                      ** FastAPI transport layer — no core logic **
  main.py                   health, meta, answer, chunk, corpus, evaluation
ui/                       ** Streamlit demo client — HTTP only, imports no pipeline **
  app.py                    entry point, navigation
  api_client.py             the only place the UI talks to the backend
  theme.py · components.py  design system and render primitives
  views/                    one module per page
generation/               ** evidence -> cited answer (see docs/generation.md) **
  safety.py                 pre-retrieval patient-specific gate
  pipeline.py               retrieve -> threshold -> prompt -> generate -> validate
  prompts.py · parsing.py · providers.py · refusal.py · schema.py · validator.py
notebooks/                notebooks only; the pipeline modules moved to the packages above
  final_evaluation.ipynb    ** the presentation notebook, 23 sections **
  build_final_notebook.py   generates final_evaluation.ipynb
eval/                     gold standards, evidence, and the frozen artifacts
  gold_standard.json          10 questions   (frozen)
  gold_standard_heldout.json  18 questions   (frozen)
  gold_standard_final20.json  20 questions   (frozen; SHA in .sha256)
  final_evidence.json         per-query retrieved evidence (pre-promotion index)
  final_evaluation_results.json
  experiment_history.json     23 experiments
  scripts/                    every evaluation / audit / integrity tool
    evaluate.py                 the frozen, retriever-agnostic evaluator
  generation/                 the generation eval: set, results, report, citation review
  runs/                       every historical run, preserved
    final_corrected_v1_final20.json  evidence for the SHIPPED config
vectordb/                 ** production vector database (Qdrant) — infrastructure only **
  config.py                 env-driven settings; no credential in code
  schema.py                 collection schema, deterministic point IDs, ingest validation
  ingest.py                 migrate the existing 991 vectors into Qdrant
  retriever.py              dense cosine top-10 from Qdrant (no rerank, no LLM)
  verify_migration.py       local vs Qdrant equivalence -> eval/qdrant_migration_verification.json
  benchmark.py              latency + footprint -> eval/qdrant_performance.json
pyproject.toml            installs the packages: pip install -e .
docker-compose.yml        local Qdrant only; the project itself is not containerised
docs/                     HANDOFF.md ** start here ** · vector_database.md · generation.md
                          experiment_history.md · presentation_story.md
                          PROJECT_B_LESSONS.md · limitations.md · deployment_readiness.md
tests/                    156 tests (29 chunking + 12 index binding + 40 vectordb + 75 generation)
aaa-clinical-ragnour/     Project B - EXTERNAL read-only reference. Not a dependency,
                          not published to git.
aaa-clinical-raggehad/    A separate person's project variant. Untouched, not published to git.
```

## Reproduce

```bash
pip install -e .                                       # installs ingestion/ retrieval/ generation/ vectordb/

python eval/scripts/evaluate.py --label baseline       # shipped config, original 10
python eval/scripts/run_final_evaluation.py            # baseline vs V1 vs V2, all 3 sets (~45 min CPU)
python eval/scripts/rescore_conservative.py            # removes the page-overlap confound (instant)
python eval/scripts/audit_corpus_and_questions.py      # corpus + question audits
python eval/scripts/run_stability_checks.py            # 16 stability / readiness checks
python eval/scripts/verify_integrity.py                # 19 integrity checks
python eval/scripts/rebuild_shipped_index.py           # rebuild data/chunks + data/embeddings
python eval/scripts/build_experiment_history.py        # regenerates history JSON + markdown
python -m pytest tests -q                              # 156 tests
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
integrity checks pass). A production vector store (Qdrant) now backs retrieval, verified
equivalent to the local index on all 48 frozen questions — but there is still no service,
no logging, no authentication, no answer generation, and **no abstention threshold** — an
out-of-scope query still returns 10 chunks. See `docs/deployment_readiness.md`.

## Project B

`aaa-clinical-ragnour/` (renamed from `aaa-clinical-rag/`) is a **separate** project kept as read-only comparison evidence. Nothing in
it was modified or re-run, and **the final project does not depend on it in any way**.

Its chunking idea was audited, isolated and tested — that is where V1 comes from. Its retrieval
machinery was refused: 10 intents for 10 evaluation questions, boosts naming specific answers by
ID and literal sentence, and a checker whose "correct answers" are **9 of 10** literal strings
copied from its own ranker's scoring rules. Measured by running its code: intents fire 10/10 on
its own questions and **1/18** on ours. It also silently truncates **143 of 452** indexed chunks.
Full audit: `eval/project_b_comparison.json`, `docs/PROJECT_B_LESSONS.md`.
