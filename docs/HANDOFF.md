# Handoff

> ⚠️ Clinical information **retrieval** prototype. Not clinically validated, not for clinical
> use. It surfaces guideline passages — including genuinely conflicting ones — and does not
> generate advice.

**Frozen at commit `47f25bb`** (`47f25bbadc8731f8859a22a42c9b6c3b892f5943`), which merges
`924beca` (P1–P4: one chunker implementation, index bound to its chunk set by content digest)
with `931847d` (Qdrant production vector store + migration verification).

## Current state

- **Retrieval research is finished and frozen.** The evaluation was run, decided and published;
  do not re-run it to replace the numbers. New work gets **new** artifacts.
- **V1 atomic (page-safe)** is the selected chunking strategy — `ingestion/atomic_chunking.py`.
  There is now exactly **one** implementation of it; the variants that used to be a forked copy in
  `eval/scripts/experimental_atomic_chunking.py` are parameters on that one function.
- **Embedding is frozen**: `abhinand/MedEmbed-base-v0.1`, revision
  `7a90c50263f620dff743eb9794b89a42bfc5d765`. Every load path applies the pin —
  `SentenceTransformer(...)` is constructed in exactly one place,
  `ingestion.chunking.load_embedder`.
- **The index is bound to the chunk set it was built from by content digest**, and the loader
  refuses to open an index whose binding no longer holds. See *Index integrity* below.
- **Qdrant is the production vector store** (collection `aaa_clinical_v1`), **verified equivalent**
  to the local numpy index on all 48 frozen questions.
- The local index (`data/embeddings/`) is **preserved unchanged** as the reproducibility artifact
  and as the reference Qdrant is checked against.
- There is **no LLM, no API service, no deployment, and no abstention threshold** yet.

## Architecture

```
4 guideline PDFs                     data/pdfs/
  │  PyMuPDF extraction, cleaning
  ▼
data/processed/                      pages.json · pages_df.parquet · recommendations.json
  │  ingestion.chunking.run_chunking(strategy="atomic")
  │    → ingestion.atomic_chunking.build_chunks   (structural anchors, page-safe,
  │                                               token-budgeted, full provenance)
  ▼
data/chunks/chunks.json              1,760 chunks produced
  │  ingestion.chunking.validate_chunks → embeddable_chunks
  │    (drops references / TOC / title-only / boilerplate / non-guideline;
  │     refuses anything over the 512-token window rather than truncating)
  ▼
991 indexed chunks
  │  retrieval.index.build_embeddings — MedEmbed-base-v0.1 @ 7a90c502, L2-normalised
  ▼
data/embeddings/                     embeddings.npy   991 × 768 float32
                                     embedded_chunks.json   991 payload records
                                     ids.json          991 ordered chunk_ids + digest
                                     index_meta.json   model, revision, dim, digests
  │
  ├─► retrieval.index.load_index → retrieve   exhaustive numpy cosine, top-10
  │       the reproducibility reference; what the frozen evaluation was scored on
  │
  └─► vectordb/ingest.py --recreate           a COPY, not a rebuild
          │   validate_index_bundle (schema contract)
          │   point_id = uuid5(NAMESPACE, chunk_id)   deterministic, never random
          ▼
      Qdrant collection `aaa_clinical_v1`    991 points, 768-dim, Cosine, exact
          │
          └─► vectordb/retriever.QdrantRetriever.search()   dense cosine, top-10
                  same hit keys as retrieval.index.retrieve, so
                  eval/scripts/evaluate.py accepts them unchanged
```

Nothing in the retrieval path branches on *which* question is being asked. No reranking, no
query rewriting or expansion, no intent detection, no keyword bonus, no filtering.

## Index integrity: how chunk_id → vector is enforced

The join from a vector to its chunk is **positional** on both sides — row *i* of
`embeddings.npy` is record *i* of `embedded_chunks.json`, and Qdrant points are built by pairing
`records[i]` with `vectors[i]`. A length check cannot detect a length-preserving change, so the
binding is asserted by content instead:

| layer | enforced by | what it asserts |
|---|---|---|
| local index | **`retrieval.index.verify_index_binding`** | `ids.json` equals the indexed `chunk_id` list **element-wise**; that list's digest equals `index_meta.indexed_chunk_ids_sha256`; `data/chunks/chunks.json` still hashes to `index_meta.source_chunks_sha256`. Raises `RuntimeError` on any mismatch — the loader refuses to return a mis-joined index. |
| Qdrant identity | `vectordb/schema.point_id_for` | `point_id = uuid5(6f1f2b9a-…, chunk_id)` — same `chunk_id` gives the same point ID on every machine, forever. The `chunk_id` is also stored verbatim in the payload. |
| Qdrant contract | `vectordb/schema.validate_index_bundle` | dimension, model, revision, metric, L2 norms, record/vector counts, no duplicate `chunk_id`, no point-ID collision, no null/NaN in required fields, no token-limit violation. Runs **before** anything is written. |
| Qdrant ↔ local | `vectordb/verify_migration.py` | one query vector down both paths, 48 frozen questions: same top-1, same top-10 set, same order, scores within 1e-5, zero metadata mismatches. |

`verify_index_binding` is called by `retrieval.index.load_index` (and therefore by
`vectordb/verify_migration.py`, `eval/scripts/evaluate.py`,
`eval/scripts/experimental_phase7_heldout.py` and notebook 03), and directly by
`eval/scripts/experimental_atomic_chunking.load_production_index`,
`eval/scripts/verify_integrity.py` and `eval/scripts/run_stability_checks.py`.

**Known gap:** `vectordb/ingest.py` uses its own `load_local_index()` rather than
`retrieval.index.load_index`, so the ingestion path — the only one that *writes* to production — is
the one path that does **not** call `verify_index_binding`. Its docstring's claim to load "exactly
as `retrieval.index.load_index` does" is not accurate. In practice the binding is covered because
`verify_migration.py` loads through `load_index` and must be run after ingestion, but that ordering
is convention, not enforcement. Closing it is a two-line change in `ingest.load_local_index`.

## Validated numbers

| | |
|---|---|
| chunks produced | **1,760** |
| indexed vectors | **991** |
| dimensions | **768** |
| similarity | **cosine** (L2-normalised, exhaustive/exact) |
| top-K | **10** |
| local vs Qdrant equivalence | **48 / 48** — same top-1, same top-10 set, same order, max score diff **2.36e-07** (tolerance 1e-05), 0 metadata mismatches, verdict **EQUIVALENT** |
| final20 **P@1** | **0.55** |
| final20 **MRR** | **0.6642** |
| final20 Recall@10 | 0.7833 |
| final20 Relevant_Top1 / Answering@5 | 11/20 · 16/20 |
| decision | **ADOPT WITH CAVEATS** |
| tests | **156 passing** (29 chunking + 12 index binding + 40 vector database + 75 generation) |
| integrity gate | **19 pass / 1 fail** — the single failure is the known `final20` freeze-hash limitation (`docs/REFERENCE_COMPARISON.md` §9) |

Evidence: `eval/final_evaluation_results.json`, `eval/runs/final_corrected_v1_final20.json`,
`eval/runs/p1_shipped_chunker_all_sets.json`, `eval/qdrant_ingestion_report.json`,
`eval/qdrant_migration_verification.json`, `eval/integrity_report.json`,
`eval/final_artifact_hashes.json` (SHA-256 of 38 tracked artifacts + 28 historical runs; if one
stops matching, something was modified after the freeze).

## Freeze record

SHA-256 of the artifacts the retrieval claims rest on, as frozen at `47f25bb`. The generated
authority is `eval/final_artifact_hashes.json`; these are the four that matter most.

| artifact | SHA-256 |
|---|---|
| `data/chunks/chunks.json` | `82faf28fb81b599f5a6ad4bcda5845d19963e2e46790f0357b1d52dc3952f0d2` |
| `data/embeddings/embeddings.npy` | `7e18ccd016438274f1e6508c051d713e6dc2def27b5a5dd0336990c102a9d8da` |
| `data/embeddings/embedded_chunks.json` | `c08f08372fba13077191dbcdee966c6ecce6bb14de7028132611ec4308e06007` |
| `data/embeddings/ids.json` | `c34155e45606c10e8ccce35f877fe3196a59e8d283ccfca444b3ff71ee3e1fae` |
| `data/embeddings/index_meta.json` | `e4de343c238c4bdebf13c16e38d1d44a00d3adc2f91c31212df59e0df4bbb6c0` |

The index's own binding record, inside `index_meta.json`:

| field | value |
|---|---|
| `source_chunks_sha256` | `82faf28fb81b599f…` — equals `chunks.json` above, so the index is provably built from it |
| `indexed_chunk_ids_sha256` | `b48e5ec5a3164d10…` — digest of the 991 ordered `chunk_id`s |

## The heldout18 metric change, and why final20 is unaffected

Three `heldout18` metrics moved **upward** when the V1 rows were recomputed on the shipped
chunker. This is a correction to a measurement, not a change to the system.

**Cause.** The published V1 rows were originally produced by `eval/scripts/experimental_atomic_chunking.py`,
which carried a *second, independent copy* of the atomic chunker. The two drifted: the copy emitted
**1,764 chunks / 1,004 indexed** where the shipped module emits **1,760 / 991**. The difference is
the citation-heading fix — numbered *bibliography* lines were being accepted as section headings —
which was off when those rows were written. P1 deleted the copy and made the variants parameters on
the shipped function, so the evaluated chunk set and the shipped chunk set cannot diverge again.

**Effect.**

| dataset | outcome |
|---|---|
| `original10` | identical in all 8 metrics |
| **`final20`** | **identical in all 8 metrics (Δ 0.0000)** |
| `heldout18` | 3 of 8 moved, all upward |

| heldout18 metric | as first published | shipped chunker | Δ |
|---|---:|---:|---:|
| P@3 | 0.5556 | **0.5741** | +0.0185 |
| P@5 | 0.4444 | **0.4556** | +0.0112 |
| MRR | 0.8148 | **0.8241** | +0.0093 |

P@1, Recall@5, Recall@10, Relevant_Top1 and Answering@5 are unchanged.

**The whole delta is one question.** Per-question comparison shows 17 of 18 questions identical.
Only **Q7 — "What cardiac assessment is needed before aneurysm repair?"** moved: its first
relevant hit went from rank 3 to rank 2, and its relevant count in the top 5 went from 2 to 3. The
arithmetic closes exactly:

- MRR: (1/2 − 1/3) / 18 = +0.00926 → **+0.0093**
- P@3: (1/3) / 18 = +0.0185 → **+0.0185**
- P@5: (1/5) / 18 = +0.0111 → **+0.0112**

**Why `final20` is untouched — the same reasoning as `docs/REFERENCE_COMPARISON.md` §4b.** The
frozen relevance rule is keyed on **document + page range**, never on `chunk_id`: a chunk counts as
relevant if its text satisfies the pre-registered fact groups *and* its page range overlaps a
pre-registered answer passage. The citation-heading fix only removed anchors that fell inside
*bibliography* text, which the content classifier already excludes from the index. So the fix
changes which chunk **ids** exist and how narrative text is split around reference sections, but it
does not move the **pages** the evidence comes from. Renumbered chunk ids drawn from the same pages
score identically — which is exactly what happened to seven other `heldout18` questions, and to
every `final20` question. `final20` therefore shows Δ 0.0000 across all eight metrics, reproducing
`eval/runs/final_corrected_v1_final20.json` byte-for-byte; that earlier gate is used as the
**control** for the recomputation, so if `final20` had moved, the recomputation itself would be
suspect.

The superseded `heldout18` row is retained and labelled in `README.md` rather than deleted, per the
append-only convention in `docs/REFERENCE_COMPARISON.md` §8b.

## Getting started

```bash
pip install -e .                         # installs ingestion/ retrieval/ generation/ vectordb/
cp .env.example .env                     # never commit .env

docker compose up -d                     # local Qdrant v1.19.0 on :6333
python vectordb/ingest.py --recreate     # load the 991 existing vectors (~1 s)
python vectordb/verify_migration.py      # must print EQUIVALENT, 48/48
python -m pytest tests -q                # 156 tests
python eval/scripts/verify_integrity.py  # 19 pass / 1 known fail
```

Query the production store:

```python
from vectordb.retriever import QdrantRetriever
hits = QdrantRetriever().search("When is elective AAA repair recommended?", top_k=10)
# chunk_id, text, score, document, page_start, page_end, section, recommendation_id, ...
```

## Next task: the answer layer

Everything below is unbuilt and is the next piece of work:

1. LLM generation over the retrieved evidence;
2. evidence-grounded answering — every claim traceable to a retrieved chunk;
3. citations (`chunk_id`, document, page, section are already on every hit);
4. **abstention** — an out-of-scope question currently still returns 10 chunks;
5. generation evaluation (faithfulness / grounding), kept separate from the frozen retrieval
   evaluation and scored on its own artifacts;
6. FastAPI service in front of `vectordb.retriever`;
7. deployment (Qdrant Cloud is already supported via `QDRANT_URL` + `QDRANT_API_KEY`).

The corpus contains **conflicting** recommendations across guidelines. The retriever surfaces
them without reconciling; the answer layer must not paper over that.

## Do not change

- **retrieval** — dense cosine, top-10, no reranking, no query rewriting/expansion, no intent
  detection, no keyword bonus, no per-question rules. Nothing in the retrieval path may branch
  on *which* question is being asked;
- **chunking** — V1 boundaries and chunk IDs;
- **embeddings** — model, pinned revision, dimension, normalisation, or the stored vectors;
- **the index binding** — `ids.json`, the digests in `index_meta.json`, and
  `verify_index_binding`. If the index is rebuilt, all of them are regenerated together by
  `eval/scripts/rebuild_shipped_index.py`; never hand-edit one of them to make a check pass;
- **Qdrant retrieval semantics** — collection config (768/Cosine/exact), the deterministic
  point-ID mapping, and the payload contract;
- **corpus** — the four guideline PDFs are frozen; no new documents, no web scraping
  (`eval/corpus_audit.json`: all 48 questions are answerable from the current index);
- **existing evaluation artifacts** — gold standards, `eval/final_*`,
  `eval/experiment_history.json`, `eval/runs/*`. New work gets **new** artifacts.

If a change to any of the above is genuinely needed, it is an experiment: run it against the
frozen gold standards, record it as a new run in `eval/runs/`, and keep the published numbers
as they are.

`eval/scripts/run_final_evaluation.py` is **not** a re-runnable report. Its `baseline_production`
configuration reads whatever index is in `data/embeddings/`, which is now V1 — so re-running it
overwrites the historical page-buffer baseline with numbers identical to V1 and destroys the
comparison the published table rests on. Recompute V1 with
`eval/scripts/run_p1_shipped_chunker_eval.py`, which writes a separate artifact and overwrites nothing.

## Where to read next

| file | what it gives you |
|---|---|
| `README.md` | results, corpus, layout, how to reproduce everything |
| `docs/vector_database.md` | Qdrant architecture, collection config, payload schema, equivalence check, latency |
| `docs/REFERENCE_COMPARISON.md` | stage-by-stage comparison against a reference RAG service; §6 P1–P9 (what was adopted and why), §8b/§8c (append-only and hash conventions), §9 the `final20` freeze-hash limitation |
| `eval/README.md` | the frozen evaluation methodology and relevance rule |
| `docs/limitations.md` · `docs/deployment_readiness.md` | what this system cannot do, and what is missing for production |
| `docs/PROJECT_B_LESSONS.md` | why the audited reference implementation's retrieval machinery was refused — read before reusing anyone else's ranking logic |
| `notebooks/final_evaluation.ipynb` | the presentation notebook, already executed |

`aaa-clinical-ragnour/`, `aaa-clinical-raggehad/` and `for ground truth code/` are other people's
projects that live in the working directory. They are git-ignored, are **not** dependencies, and
nothing in this project imports them.
