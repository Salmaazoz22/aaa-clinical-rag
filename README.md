# AAA Clinical RAG

> ⚠️ **Research prototype — not clinically validated, not for clinical use.**
> Retrieves guideline passages and generates structured cited answers, but makes no claim of
> clinical safety, accuracy, or fitness for patient care. The corpus contains genuinely conflicting
> recommendations across guidelines, which the system surfaces without reconciling.

A retrieval-augmented generation system over four abdominal aortic aneurysm (AAA) clinical
guidelines — USPSTF 2019, NICE NG156, ESVS 2024, and SVS 2018. Questions are answered with
structured citations, validated against the evidence actually sent to the model, and refused
when evidence is insufficient.

&nbsp;

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/291bb882-5e4f-48bd-bedf-3b3b26f317cf" />

*Clinical evidence retrieval and voice-enabled question interface*

&nbsp;

## Features

- **Cited, validated answers** — every response cites specific guideline chunks; a citation
  validator checks each reference against what was actually sent to the model
- **Layered safety gates** — emergency detection, patient-specific blocking, guideline-scope
  checking, evidence-quality floor, and model-level refusal
- **Cross-provider fallback** — primary LLM failure automatically retries on a secondary
  provider under a bounded wall-clock deadline (default 90 s)
- **Frozen retrieval** — dense cosine similarity only; no query rewriting, no intent
  detection, no per-question rules
- **Full audit trail** — retrieval scores, safety verdicts, validator findings, stage timings,
  and provider metadata on every response
- **Voice input** — Whisper speech-to-text for question composition
- **524 tests** — chunking, vector DB, generation, citation validation, safety gates, API,
  UI isolation, and provider fallback

&nbsp;

## How It Works

```
Browser → Streamlit (ui/) → HTTP → FastAPI (api/) → generation.pipeline.answer_question
                                                     ├─ safety gates (emergency, patient, scope)
                                                     ├─ MedEmbed → Qdrant (retrieval)
                                                     ├─ evidence threshold
                                                     ├─ LLM (Groq / OpenRouter)
                                                     └─ citation validator
```

**API** — thin transport layer; calls `answer_question` and serialises the result.
No core logic lives here.

**UI** — pure HTTP client. Nothing under `ui/` imports `generation`, `retrieval`, `vectordb`,
or `ingestion` — enforced by test.

&nbsp;

Every question goes through four key processing stages: safety screening, dense evidence retrieval, thresholding, and model generation with mandatory citation validation.

### Evidence Retrieval

Relevant passages are retrieved from indexed guideline chunks using dense cosine similarity. Each retrieved chunk includes metadata enabling direct verification against the original source document.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/8d84ca67-b906-47a9-a2fe-be362b962095" />

*Evidence retrieval with source-level guideline verification*

&nbsp;

### Grounded Answer Pipeline

Qualifying evidence is formatted into a structured prompt contract. The model generates an answer with explicit citations, which are validated against the evidence chunks actually provided.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/6184a141-2ca3-41cc-ba79-76ba12fab9a3" />

*End-to-end retrieval, grounding, and citation validation*

&nbsp;

## Safety and Reliability

Every question passes through a chain of safety gates before an answer is produced.
All gates except the last are deterministic — no model call required.

| Gate | What it catches | Outcome |
|---|---|---|
| Emergency | Acute symptoms + AAA context | Redirect to emergency services |
| Patient-specific | Diagnosis/dosing for a named individual | Refuse locally — details never leave the machine |
| Guideline scope | Edition not in the corpus | Refuse with available editions |
| Evidence floor | No chunk clears the similarity threshold | Refuse, citing what was examined |
| Model judgement | Model considers evidence insufficient | Recorded as model-level refusal |

When retrieved evidence does not meet the configured similarity threshold, the system refuses
rather than generating a clinical answer from weak or merely topically related evidence.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/aa093896-f2fb-45b6-8c95-5976861a5165" />

*Evidence-threshold abstention when retrieved evidence is insufficient*

&nbsp;

When retrieved chunks from different guidelines disagree, the answer surfaces both positions
with their own citations rather than silently resolving the conflict.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/29319eb9-891e-48e9-8428-05a88fd5cceb" />

*Guideline disagreement detection and citation validation*

&nbsp;

**Cross-provider fallback:** when the primary provider fails (rate limit, timeout, dead model
slug), the same request retries on the secondary. Both providers operate under a single
wall-clock deadline so the caller always gets a bounded response time. Both API keys must be
set for fallback to activate.

Details: `docs/generation.md`

&nbsp;

## Quick Start

**Prerequisites:** Python ≥ 3.10, Docker, an API key for Groq and/or OpenRouter.

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env              # edit .env — set GROQ_API_KEY at minimum

# 3. Start Qdrant
docker compose up -d              # v1.19.0 on :6333 (REST) / :6334 (gRPC)

# 4. Populate the vector store
python vectordb/ingest.py --recreate
python vectordb/verify_migration.py

# 5. Run
uvicorn api.main:app --port 8000  # terminal 1 — API (docs at /docs)
streamlit run ui/app.py           # terminal 2 — UI at localhost:8501
```

&nbsp;

## Configuration

All settings are environment-driven via `.env` (git-ignored). No credential is ever hardcoded.
See `.env.example` for the full reference with documentation.

**Essential variables:**

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Primary LLM provider key |
| `OPENROUTER_API_KEY` | — | Fallback provider key (strongly recommended) |
| `GENERATION_PROVIDER` | `groq` | Which provider to use (`groq` / `openrouter`) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `aaa_clinical_v1` | Collection name |

**Tuning and reliability:**

| Variable | Default | Purpose |
|---|---|---|
| `GENERATION_SCORE_THRESHOLD` | `0.75` | Evidence-quality floor (cosine similarity) |
| `GENERATION_TOP_K` | `5` | Chunks retrieved per question |
| `GENERATION_ENABLE_FALLBACK` | `true` | Cross-provider retry on failure |
| `GENERATION_DEADLINE` | `90` | Wall-clock budget for the fallback chain (seconds) |
| `GENERATION_TIMEOUT` | `60` | Per-provider socket timeout (seconds) |

Provider model slugs are pinned in code and overridable via `GROQ_MODEL`, `OPENROUTER_MODEL`,
or `GENERATION_MODEL`. If a vendor retires a model, it's a `.env` edit, not a code change.

Full variable reference: `.env.example` · Provider details: `docs/generation.md`

&nbsp;

## Usage

**CLI:**

```bash
# Answer a question
python -m generation.pipeline "What diameter threshold triggers elective AAA repair?"

# Full JSON audit record
python -m generation.pipeline "..." --json

# With prompt and overrides
python -m generation.pipeline "..." --json --show-prompt --provider openrouter --top-k 3

# Direct vector search
python vectordb/retriever.py "When is elective AAA repair recommended?"
```

**API:**

| Endpoint | Description |
|---|---|
| `GET /health` | API, Qdrant, index, and LLM-key status |
| `POST /v1/answer` | Answer a question — full audit record, refusals included |
| `GET /v1/meta` | Model, revision, collection, dimensions, chunk count |
| `GET /v1/chunks/{chunk_id}` | Chunk payload for citation audit |
| `GET /v1/corpus` | Guideline documents with live chunk counts |
| `GET /v1/evaluation` | Frozen retrieval evaluation with SHA-256 digests |

```bash
curl -X POST http://localhost:8000/v1/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What diameter threshold triggers elective AAA repair?"}'
```

Errors are never dressed up as answers: missing key → `503`, unparseable model response → `502`,
invalid question → `400`.

**UI pages:** Ask · Evaluation · Safety & Abstention · Architecture · Guidelines & Sources · Technical Details

&nbsp;

## Development and Testing

```bash
python -m pytest tests -q                     # 524 tests
python -m pytest tests/test_generation.py -q  # single module
```

Tests cover chunking, index binding, vector DB, generation pipeline, citation validation,
safety gates, provider fallback, API endpoints, UI isolation, and voice transcription.
All generation tests use injectable mocks — no vector store or network call required.

**Evaluation and reproduction:**

```bash
python eval/scripts/run_final_evaluation.py        # retrieval: baseline vs V1 vs V2, all sets
python eval/generation/run_generation_eval.py      # generation: answer quality
python eval/scripts/verify_integrity.py            # integrity checks
python eval/scripts/run_stability_checks.py        # stability checks
jupyter notebook notebooks/final_evaluation.ipynb  # presentation notebook (outputs committed)
```

Full experiment history: `docs/experiment_history.md`

&nbsp;

## Corpus and Index

| Property | Value |
|---|---|
| Guidelines | USPSTF 2019 · NICE NG156 · ESVS 2024 · SVS 2018 |
| Chunks | 1,760 total → 991 indexed |
| Embedding | `abhinand/MedEmbed-base-v0.1` · 768-dim · L2-normalised |
| Token safety | Max 512 tokens · 0 over the model window |
| Vector store | Qdrant (production) · NumPy (research artifact) |
| Search | Dense cosine · exhaustive · top-K |

**Retrieval results** (the pre-registered `final20` set):

| Config | P@1 | MRR | R@10 |
|---|---:|---:|---:|
| Baseline (page-buffer) | 0.400 | 0.531 | 0.608 |
| **V1 atomic (shipped)** | **0.550** | **0.664** | **0.783** |

Same retriever in both rows — the only variable is chunk boundaries.

Full results: `eval/final_evaluation_results.json` · Details: `docs/experiment_history.md`

&nbsp;

## Project Structure

```
ingestion/        Corpus → chunks (PDF extraction, atomic chunking)
retrieval/        Embeddings, local index, optional reranker
vectordb/         Qdrant: config, schema, ingest, retriever, migration verification
generation/       Evidence → cited answer: pipeline, safety gates, providers, validator
api/              FastAPI transport layer (no core logic)
ui/               Streamlit client (HTTP only, no pipeline imports)
data/             PDFs, extracted pages, chunks, embeddings
eval/             Gold standards, frozen evaluation artifacts, scripts
notebooks/        Presentation notebook (23 sections, outputs committed)
docs/             Detailed documentation (start with HANDOFF.md)
tests/            524 tests across 16 modules
```

&nbsp;

## Deployment

The project requires two processes:

| Process | Command | Requirements |
|---|---|---|
| Backend | `uvicorn api.main:app` | ≥ 2 GB RAM, public URL, Qdrant access |
| Frontend | `streamlit run ui/app.py` | `API_URL` pointing at the backend |

The UI is deployable to Streamlit Community Cloud with `requirements-streamlit.txt` (minimal
dependencies: `streamlit`, `requests`, `openai` — no PyTorch). The backend needs the full
dependency set.

Qdrant Cloud is supported: point `QDRANT_URL` at the cluster, set `QDRANT_API_KEY`, and run
`python vectordb/ingest.py` once.

Full guide: `docs/DEPLOYMENT.md`

&nbsp;

## Documentation

| Document | Contents |
|---|---|
| `docs/HANDOFF.md` | **Start here** — project overview and handoff context |
| `docs/generation.md` | Generation layer: gates, validation, providers, evaluation |
| `docs/DEPLOYMENT.md` | Deployment guide for Streamlit Cloud + backend |
| `docs/vector_database.md` | Qdrant migration, equivalence verification, benchmarks |
| `docs/experiment_history.md` | All 23 retrieval experiments with results |
| `docs/deployment_readiness.md` | What is verified and what is not |
| `docs/limitations.md` | Known limitations and gaps |

&nbsp;

## Troubleshooting

| Problem | Fix |
|---|---|
| `503` on `/v1/answer` | Set `GROQ_API_KEY` in `.env` |
| `502` on `/v1/answer` | Set both API keys for fallback; check provider status |
| UI shows "backend unavailable" | Start the API; verify `API_URL` points to the correct address |
| Slow encoding in containers | Set `TORCH_NUM_THREADS=1` |
| `413 Request too large` (Groq) | Free-tier limit; raise `GROQ_MAX_REQUEST_TOKENS` on a paid tier |
| Fallback not activating | Both API keys must be set; check if the secondary model slug was retired |

&nbsp;

## Status

This is a **research prototype**, not a production clinical system.

✅ Deterministic, reproducible retrieval · Qdrant vector store verified against local index ·
Grounded generation with structured citations · Five safety gates · Citation validation ·
Cross-provider fallback with bounded latency · FastAPI service · Streamlit UI with voice
input · 524 tests

❌ Not clinically validated · No authentication or rate limiting · No persistent logging ·
Abstention threshold is a starting value, not calibrated · No CI-verified PDF-to-index
reproducibility

Full assessment: `docs/deployment_readiness.md`

&nbsp;

## License

See the repository for license information.
