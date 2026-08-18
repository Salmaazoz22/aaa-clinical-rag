# Handoff

> ⚠️ Clinical information **retrieval** prototype. Not clinically validated, not for clinical
> use. It surfaces guideline passages — including genuinely conflicting ones — and does not
> generate advice.

## Current state

- **Retrieval research is finished and frozen.** The evaluation was run, decided and published;
  do not re-run it to replace the numbers.
- **V1 atomic (page-safe)** is the selected chunking strategy — `notebooks/clinical_atomic_chunking.py`.
- **Embedding is frozen**: `abhinand/MedEmbed-base-v0.1`, revision `7a90c50263f620dff743eb9794b89a42bfc5d765`.
- **Qdrant is the production vector store** (collection `aaa_clinical_v1`), and is **verified
  equivalent** to the local numpy index on all 48 frozen questions.
- The local index (`data/embeddings/`) is **preserved unchanged** as the reproducibility
  artifact and as the reference Qdrant is checked against.
- There is **no LLM, no API service, no deployment, and no abstention threshold** yet.

## Validated numbers

| | |
|---|---|
| indexed vectors | **991** |
| dimensions | **768** |
| similarity | **cosine** (L2-normalised, exhaustive/exact) |
| top-K | **10** |
| local vs Qdrant equivalence | **48 / 48** — same top-1, same top-10, same order, max score diff 2.075e-07 (tol 1e-05) |
| final20 **P@1** | **0.55** |
| final20 **MRR** | **0.6642** |
| final20 Recall@10 | 0.7833 |
| decision | **ADOPT WITH CAVEATS** |
| tests | **69 passing** (29 pipeline + 40 vector database) |

Evidence: `eval/final_evaluation_results.json`, `eval/runs/final_corrected_v1_final20.json`,
`eval/qdrant_migration_verification.json`, `eval/final_artifact_hashes.json` (SHA-256 of every
frozen artifact — 63 files; if one stops matching, something was modified after the freeze).

## Getting started

```bash
pip install -r requirements.txt
cp .env.example .env                     # never commit .env

docker compose up -d                     # local Qdrant v1.19.0 on :6333
python vectordb/ingest.py --recreate     # load the 991 existing vectors (~1 s)
python vectordb/verify_migration.py      # must print EQUIVALENT, 48/48
python -m pytest tests -q                # 69 tests
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
- **corpus** — the four guideline PDFs are frozen; no new documents, no web scraping
  (`eval/corpus_audit.json`: all 48 questions are answerable from the current index);
- **existing evaluation artifacts** — gold standards, `eval/final_*`, `eval/experiment_history.json`,
  `eval/runs/*`. New work gets **new** artifacts.

If a change to any of the above is genuinely needed, it is an experiment: run it against the
frozen gold standards, record it as a new run in `eval/runs/`, and keep the published numbers
as they are.

## Where to read next

| file | what it gives you |
|---|---|
| `README.md` | results, corpus, layout, how to reproduce everything |
| `docs/vector_database.md` | Qdrant architecture, collection config, payload schema, equivalence check, latency |
| `eval/README.md` | the frozen evaluation methodology and relevance rule |
| `docs/limitations.md` · `docs/deployment_readiness.md` | what this system cannot do, and what is missing for production |
| `docs/PROJECT_B_LESSONS.md` | why the audited reference implementation's retrieval machinery was refused — read before reusing anyone else's ranking logic |
| `notebooks/final_evaluation.ipynb` | the presentation notebook, already executed |

`aaa-clinical-ragnour/` and `aaa-clinical-raggehad/` are other people's projects that live in
the working directory. They are git-ignored, are **not** dependencies, and nothing in this
project imports them.
