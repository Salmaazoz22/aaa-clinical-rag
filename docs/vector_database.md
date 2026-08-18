# Production Vector Database (Qdrant)

The retrieval research is finished and frozen. This document describes the step that follows
it: putting the **already-validated** index behind a real vector database, and proving that
doing so changed nothing about what the system retrieves.

> **Nothing in the validated pipeline was touched.** Same V1 atomic chunks, same chunk IDs,
> same `abhinand/MedEmbed-base-v0.1` @ `7a90c502…`, same 768 dimensions, same L2 normalisation,
> same dense cosine, same top-K = 10, same 991 indexed chunks, same gold standards, same
> published metrics. Qdrant stores the **same vectors**; it does not recompute them.

---

## Architecture

```
question
  │
  ▼
MedEmbed-base-v0.1 @ 7a90c502   query embedding, 768-d, L2-normalised   (unchanged)
  │
  ▼
Qdrant  ── collection `aaa_clinical_v1`, 991 points, cosine, exact search
  │
  ▼
top-10 evidence  (chunk_id, text, score, document, pages, section, recommendation_id)
  │
  ▼
[ LLM answer layer — NOT built yet ]
```

Two back ends now exist, deliberately:

| | local numpy index | Qdrant |
|---|---|---|
| Path | `data/embeddings/embeddings.npy` + `embedded_chunks.json` | collection `aaa_clinical_v1` |
| Role | **research artifact** — what every published number was produced from | **production retrieval backend** |
| Used by | `eval/*`, `notebooks/*`, the frozen evaluation | the service layer (FastAPI, later) |
| Status | preserved, unchanged, still reproducible | derived copy, rebuildable at any time |

The local index is not a fallback that drifts — it is the reference the production store is
verified *against*. If the two ever disagree, the local index wins and the migration is wrong.

---

## Collection configuration

| Setting | Value | Why |
|---|---|---|
| Collection | `aaa_clinical_v1` (`QDRANT_COLLECTION`) | one collection for the frozen guideline corpus |
| Vector size | **768** | MedEmbed-base-v0.1 output dimension; ingestion fails on anything else |
| Distance | **Cosine** | the validated similarity function |
| Points | **991** | the indexed clinical chunks, 1:1 with `embedded_chunks.json` |
| Search | `exact=true` | the validated retriever is exhaustive; HNSW approximation is not used |
| Point ID | `uuid5(6f1f2b9a-6a6b-5c4d-9e3f-aa0c11e01a01, chunk_id)` | deterministic, never random |

**Vectors are stored, not regenerated.** `vectordb/ingest.py` reads `embeddings.npy` and uploads
those exact float32 values. Qdrant re-normalises on insert for cosine, which is a no-op here
because the vectors are already L2-normalised (checked at ingest, tolerance 1e-3).

**Why `exact=true`.** Qdrant only builds an HNSW graph past its indexing threshold (20 000
points by default), so 991 points are already searched exhaustively. The flag makes that
explicit and keeps it true if the corpus ever grows — an approximate index would be a change
to retrieval semantics, and this migration is not allowed to make one.

### Deterministic point IDs

Qdrant point IDs must be an unsigned integer or a UUID, and `ESVS_2024__p1-1__c0001` is
neither. The chunk ID is therefore mapped through UUIDv5 under a fixed namespace:

```python
point_id = uuid5(UUID("6f1f2b9a-6a6b-5c4d-9e3f-aa0c11e01a01"), chunk_id)
# ESVS_2024__p1-1__c0001 -> e6c01b73-2cae-5f18-b3df-8f188e609edd
```

Same chunk ID → same point ID, on every machine, on every re-ingest, forever. Re-running
ingestion overwrites points in place rather than duplicating them. The original `chunk_id` is
stored verbatim in the payload and is what every downstream artifact refers to; the UUID is an
implementation detail of the store. The mapping is pinned by a test, because changing the
namespace would silently orphan all 991 points.

---

## Payload schema

Every point carries the **complete** indexed record, so evidence can be reconstructed from
Qdrant alone without reading any local file:

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | str | the project's identity for the chunk, e.g. `ESVS_2024__p22-22__c0238` |
| `chunk_text` | str | the exact text that was embedded |
| `document_id` | str | `ESVS_2024`, `NICE_NG156`, `SVS_2018`, `USPSTF_2019` |
| `document_name` | str | full title as printed |
| `document_type` | str | e.g. `official guideline` |
| `is_guideline` | bool | always true for indexed chunks |
| `source_file` | str | source PDF filename |
| `page_number` | int | primary page |
| `page_start`, `page_end` | int | page span, used by the frozen relevance rule |
| `section_title` | str \| null | detected section heading |
| `section_source` | str | how the section was resolved |
| `content_type` | str | always `clinical` for indexed chunks |
| `token_count` | int | ≤ 512, validated at ingest |
| `char_count` | int | |
| `source_excerpt` | str | provenance excerpt from the page text |
| `recommendation_id` | str \| null | e.g. `Recommendation 24` where the chunk is a graded recommendation |
| `recommendation_grade` | str \| null | |
| `evidence_level` | str \| null | |

Payload completeness is enforced, not hoped for: a missing field, or a null in any of
`chunk_id`, `document_id`, `document_name`, `document_type`, `content_type`, `token_count`,
`page_number`, `page_start`, `page_end`, `source_file`, `chunk_text`, aborts the ingestion.

### One documented representation difference: `NaN` → `null`

`data/embeddings/embedded_chunks.json` stores some optional fields as the JSON literal `NaN`
rather than `null` — pandas missing values that survived serialisation when the frozen index
was written:

| field | NaN values |
|---|---:|
| `recommendation_id` | 113 |
| `recommendation_grade` | 161 |
| `evidence_level` | 135 |

`NaN` and `null` mean the same thing here ("this chunk carries no such value"), and Qdrant
stores `NaN` as `null`. The migration therefore maps `NaN → null` **in nullable fields only**,
and *counts every occurrence* in `eval/qdrant_ingestion_report.json` and
`eval/qdrant_migration_verification.json` (`nan_normalised_to_null`: 90 occurrences across the
48 verification queries) so the conversion is visible rather than silent. `NaN` in a required
field, or anywhere in a vector, is still a hard failure. The frozen artifact itself was **not**
edited.

---

## Ingestion

```bash
docker compose up -d                     # local Qdrant on :6333
python vectordb/ingest.py --recreate     # migrate the 991 vectors
```

`--recreate` is required to touch a collection that already exists; without it, ingestion
refuses rather than overwriting a populated production collection by accident.

The script fails loudly, before writing anything, if:

- the vector dimension is not 768, or the shape is wrong;
- any vector contains NaN or Inf, or is not L2-normalised within 1e-3;
- the vector count does not equal the chunk count, or is not the expected 991;
- `index_meta.json` disagrees with the vectors on disk;
- any chunk ID is missing, empty, or duplicated;
- two chunk IDs collide onto one point UUID;
- any payload field is missing, or a required field is null/NaN;
- any chunk exceeds the 512-token embedding window;
- `index_meta.json` does not name the pinned model, revision and cosine metric.

Nothing is repaired, coerced or dropped. Bad data is fixed in the pipeline that produced it,
not on the way into the database.

After upload it re-reads the live collection and checks vector size, distance and exact point
count, then writes `eval/qdrant_ingestion_report.json`.

**Measured:** 991 points in **0.95 s** (~1 046 points/s) over REST to local Docker.

---

## Retrieval

```python
from vectordb.retriever import QdrantRetriever

hits = QdrantRetriever().search("What diameter threshold triggers elective AAA repair?", top_k=10)
```

or from the shell:

```bash
python vectordb/retriever.py "When should elective repair be offered?" --top-k 10
```

Hits use the **same keys** as `clinical_rag.retrieve`, so existing consumers (including
`eval/evaluate.py`) accept them unchanged, plus `text` / `score` aliases and `point_id`:

```
rank, similarity_score, score, chunk_id, document, document_id, document_type, is_guideline,
section, content_type, token_count, page, page_start, page_end, source_file, source_excerpt,
chunk_text, text, recommendation_id, recommendation_grade, evidence_level, point_id
```

No reranking. No query rewriting or expansion. No intent detection. No keyword bonus. No
metadata filtering. No per-question branch. No LLM. The retriever cannot see *which* question
it is answering — which is the property that makes the frozen evaluation meaningful, and it is
preserved exactly.

Malformed input is refused rather than guessed at: empty/whitespace queries, non-string
queries, queries over 20 000 characters, non-positive or non-integer `top_k`, and query vectors
of the wrong dimension all raise.

---

## Local vs Qdrant equivalence

```bash
python vectordb/verify_migration.py     # -> eval/qdrant_migration_verification.json
```

All 48 frozen questions (original10 + heldout18 + final20) are embedded **once** each, and the
single query vector is sent down both paths — numpy exhaustive cosine, and Qdrant exact cosine.
Embedding once removes the encoder as a variable, so any difference is attributable to storage,
distance computation, ID mapping or index configuration, which is what the migration changed.

The question sets are used here only as a fixed, non-cherry-picked query sample. **No metric is
computed, no gold standard is scored, nothing is tuned.** This is an infrastructure equivalence
check, not an evaluation, and it lives in its own artifact.

### Result

| check | result |
|---|---|
| queries compared | **48 / 48 pass** |
| identical top-1 chunk ID | **48 / 48** |
| identical top-10 chunk ID set | **48 / 48** |
| identical ranking order | **48 / 48** |
| max abs. score difference | **2.075 × 10⁻⁷** (tolerance 1 × 10⁻⁵) |
| document / page / section mismatches | **0** |
| verdict | **EQUIVALENT** |

The residual ~2 × 10⁻⁷ is float32 accumulation order in the dot product, four orders of
magnitude below the tolerance and far below any score gap that could reorder a result.

Pass requires: same top-1, same top-10 set, all scores within tolerance, no metadata mismatch,
and identical order (a swap between two chunks whose scores are equal within tolerance counts
as a tie, not a disagreement — none occurred). If a run ever fails, the script exits non-zero
and prints the investigation list: normalisation, distance metric, vector precision, ID
mapping, filtering, metadata, query embedding, index configuration. **The system must not be
tuned to make the comparison pass.**

---

## Performance

```bash
python vectordb/benchmark.py            # -> eval/qdrant_performance.json
```

48 questions × 3 repeats = 144 measurements, CPU only, local Docker over REST, after 3 warm-up
queries:

| stage | mean | p95 |
|---|---:|---:|
| query embedding (MedEmbed, CPU) | 39.5 ms | 49.6 ms |
| **Qdrant search** | **9.8 ms** | 26.3 ms |
| end to end (embed + search) | 51.1 ms | 66.1 ms |
| local numpy search (reference) | 0.4 ms | 0.5 ms |

| footprint | |
|---|---|
| points | 991 |
| collection storage | 10.0 MiB |
| container memory (idle, post-ingest) | ~92 MiB |
| raw vectors (991 × 768 × float32) | 2.9 MiB |
| ingestion | 0.95 s |

Qdrant search is ~25× slower than an in-process numpy dot product over 991 vectors — the cost
of a network hop for a corpus this small — and is still under a fifth of the embedding time,
which dominates. Nothing here is optimised; these are the measured starting values.

---

## Configuration

All settings come from the environment (or a git-ignored `.env`). No credential is hardcoded,
logged, or written to any artifact — `QdrantSettings.describe()` reports only
`api_key_supplied: true|false`.

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | server endpoint |
| `QDRANT_API_KEY` | *(unset)* | Qdrant Cloud / secured instance key |
| `QDRANT_COLLECTION` | `aaa_clinical_v1` | collection name |
| `QDRANT_PREFER_GRPC` | `false` | use gRPC instead of REST |
| `QDRANT_TIMEOUT` | `30` | client timeout, seconds |
| `QDRANT_EXACT_SEARCH` | `true` | exhaustive search instead of HNSW approximation |
| `QDRANT_LOCAL_PATH` | *(unset)* | embedded local mode (dev/tests only; no server) |

### Local development

```bash
cp .env.example .env
docker compose up -d                     # Qdrant v1.19.0, :6333 REST / :6334 gRPC
python vectordb/ingest.py --recreate
python vectordb/verify_migration.py
docker compose down                      # add -v to delete the stored collection
```

Dashboard: <http://localhost:6333/dashboard>.

### Qdrant Cloud / remote

```bash
export QDRANT_URL="https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="…"                # never commit this
export QDRANT_COLLECTION="aaa_clinical_v1"
python vectordb/ingest.py --recreate
python vectordb/verify_migration.py       # same equivalence check against the remote cluster
```

The equivalence check is the acceptance test for any deployment target: a remote cluster is
only considered correctly provisioned once it reproduces the local ranking on all 48 queries.

---

## Tests

`tests/test_vectordb.py` — 40 tests, added alongside the existing 29 (69 total, all passing).
They build a real collection through the production ingestion path and query it: embedded local
mode by default so they need no Docker, or against a live server with
`QDRANT_TEST_URL=http://localhost:6333` (both were run; both pass). The test collection is
`aaa_clinical_test` and is deleted afterwards — the production collection is never touched.

Coverage: collection creation and configuration · refusal to clobber an existing collection ·
vector dimension · vector count · deterministic and collision-free point IDs · pinned ID
mapping · payload completeness and fidelity to the source record · duplicate chunk IDs ·
missing chunk ID · missing/null/NaN fields · NaN→null accounting · token-limit violation ·
wrong dimension · NaN/Inf vectors · unnormalised vectors · swapped or unpinned model ·
retrieval order · self-retrieval · top-K behaviour · local-vs-Qdrant equivalence · the recorded
verification artifact · empty query · non-string query · over-long query · invalid `top_k` ·
wrong-dimension query vector · no API key in any describe() output.

---

## What this migration did not do

- did not re-chunk, re-embed, or change any vector;
- did not change the model, revision, dimension, normalisation, distance or top-K;
- did not add reranking, query expansion, intent detection, filtering or per-question rules;
- did not add documents or scrape anything — the corpus stays frozen;
- did not import or depend on Project B (`aaa-clinical-ragnour/`) in any way;
- did not add an LLM;
- did not modify any gold standard, historical run, evidence file or published metric.

New artifacts only: `eval/qdrant_migration_verification.json`,
`eval/qdrant_ingestion_report.json`, `eval/qdrant_performance.json`.
