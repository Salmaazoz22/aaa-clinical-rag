# Deployment — Streamlit Community Cloud

**Status: the Streamlit side is ready. The application as a whole is not deployable to
Streamlit Community Cloud alone, because the frontend is a pure HTTP client and the
FastAPI backend has to run somewhere.** This document says exactly what is prepared,
what is still required, and why the shortcut is not taken.

---

## 1. The architecture constraint, stated plainly

`ui/` contains no retrieval implementation. It has no embedding model, no vector store
handle, and no copy of the generation pipeline. Every number it shows was produced by
the FastAPI service and fetched over HTTP.

This is not incidental — it is enforced:

```
tests/test_ui.py::test_ui_never_imports_the_rag_pipeline
    parses every source file under ui/ and fails if any of them imports
    generation, retrieval, vectordb or ingestion
```

So a Streamlit deployment **cannot** answer questions by itself. Two processes are
required:

| Process | What it is | Where it can run |
|---|---|---|
| Frontend | `streamlit run ui/app.py` | Streamlit Community Cloud |
| Backend | `uvicorn api.main:app` | anywhere with ≥2 GB RAM and a public URL |

**The shortcut was not taken.** Importing `generation.pipeline` directly into the
Streamlit app would make one deployable process, and it would also delete the property
the whole UI is built on (one retrieval implementation, behind one API, with every
displayed number traceable to a service response) and break the test that guards it.
That is an architectural rewrite, not a deployment fix.

---

## 2. What is prepared on the Streamlit side

| Item | State |
|---|---|
| Entry point | `ui/app.py` — set this as the Streamlit Cloud "Main file path" |
| Native theme | `.streamlit/config.toml` — committed, includes `enableStaticServing = true` |
| Fonts and favicon | `ui/static/` — committed, so there is no external font request |
| Backend URL | read from a Streamlit **secret** first, then an environment variable, then the localhost default (`ui/api_client.base_url`) |
| Accepted setting names | `API_URL`, then `CLINICAL_RAG_API_URL` (`ui/api_client.API_URL_SETTINGS`) |
| Secrets template | `.streamlit/secrets.toml.example` |
| Secrets are ignored by git | `.streamlit/secrets.toml` is in `.gitignore` |

### The localhost coupling is resolved for the frontend

`ui/api_client.DEFAULT_BASE_URL` is `http://127.0.0.1:8000`. In a cloud container nothing
listens there, so that default is exactly wrong — which is why `base_url()` reads the
configured URL from `st.secrets` before falling back to it. Streamlit Community Cloud has
no environment-variable panel; secrets are the only channel, and reading `os.environ`
alone would have made the setting unreachable.

**The setting name is part of the contract.** The deployed app read only
`CLINICAL_RAG_API_URL` while the Streamlit secret was named `API_URL`; the lookup missed,
`base_url()` returned the localhost default, and the live UI reported *"Nothing is
answering at http://127.0.0.1:8000"* with the secret correctly set the whole time. Both
names are now accepted, `API_URL` first — see `ui/api_client.API_URL_SETTINGS`.

**Cold starts.** A hosted backend that has been idle takes time to answer its first
request: 24 s measured against the Railway deployment, against 0.4 s warm. The health
probe therefore allows `REMOTE_HEALTH_TIMEOUT` (30 s) instead of the local 8 s whenever a
URL is configured, so a sleeping container is not reported as a dead one.

Three strings in `ui/views/` still print `uvicorn api.main:app --port 8000` — these are
the "the backend is not running" help texts. They are instructions for a local operator,
not a runtime dependency, and are correct for the local workflow.

---

## 3. Required secrets and environment variables

### Frontend (Streamlit Cloud → App settings → Secrets)

```toml
API_URL = "https://aaa-clinical-rag-production.up.railway.app"
```

That is the whole list. No API key belongs in the frontend's secrets: the UI never calls
Groq, OpenRouter or Qdrant.

### Backend (wherever `uvicorn api.main:app` runs)

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | primary generation provider |
| `OPENROUTER_API_KEY` | strongly recommended | without it there is **no fallback**, and a Groq rate limit becomes a 502 |
| `GENERATION_ENABLE_FALLBACK` | no (default `true`) | cross-provider retry |
| `GENERATION_FALLBACK_MODEL` | no | set when the pinned OpenRouter slug is retired — see below |
| `QDRANT_URL` + `QDRANT_API_KEY` | yes, for Qdrant Cloud | the vector store |
| `QDRANT_COLLECTION` | yes | `aaa_clinical_v1` |
| `QDRANT_LOCAL_PATH` | alternative to the above | embedded mode against a directory; development only |
| `GENERATION_TOP_K`, `GENERATION_SCORE_THRESHOLD`, `GENERATION_TEMPERATURE`, `GENERATION_MAX_OUTPUT_TOKENS`, `GENERATION_TIMEOUT` | no | pinned defaults in `generation/config.py` |

`.env.example` documents every name. It contains no values and must stay that way.

#### On the fallback model

The OpenRouter pin is `openai/gpt-oss-20b:free` — the free sibling of the Groq primary,
same family, same OpenAI-compatible contract. It replaced
`deepseek/deepseek-r1:free`, which OpenRouter withdrew; every fallback call 404'd, which
disabled the fallback with no symptom other than 502s under load. **If that happens
again, it is a `.env` edit, not a code change:**

```
GENERATION_FALLBACK_MODEL=deepseek/deepseek-r1     # or any current OpenRouter slug
```

`GROQ_MODEL` and `OPENROUTER_MODEL` do the same per provider, and apply whether the
provider is acting as primary or as secondary.

---

## 4. The embedding model and the index

The backend loads `sentence-transformers` at startup and pins the embedding revision
(`vectordb/schema.py`). Two consequences for a deployed backend:

* **Memory.** The model plus the Qdrant client needs roughly 1.5–2 GB resident. Streamlit
  Community Cloud's ~1 GB per app is below that, which is a second, independent reason the
  backend cannot be folded into the Streamlit process.
* **The index is not in git.** `.qdrant_local/` is git-ignored on purpose. What *is*
  committed is `data/embeddings/` (vectors, ids, `index_meta.json`), from which the
  collection is rebuilt:

  ```bash
  python vectordb/ingest.py        # rebuilds aaa_clinical_v1 from data/embeddings/
  python eval/scripts/verify_integrity.py
  ```

  Point `QDRANT_URL`/`QDRANT_API_KEY` at a Qdrant Cloud cluster and run the same command
  to populate it once.

---

## 5. Dependencies

Streamlit Community Cloud installs the repository's root `requirements.txt`, which is the
**backend's** dependency set: it pulls `sentence-transformers` and `transformers`, and
with them PyTorch. The frontend needs none of that — it needs `streamlit` and `requests`.

`requirements-streamlit.txt` records the minimal frontend set. Community Cloud will not
pick it up automatically from a repo that also has `requirements.txt`; it is there for the
UI-only deployment route (a branch or repo containing `ui/`, `assets/`, `.streamlit/` and
that file). Deploying this repository as-is works, but installs far more than the UI uses
and may hit the platform's build time and image size limits.

---

## 6. Checklist

Frontend, on Streamlit Community Cloud:

- [x] entry point `ui/app.py`
- [x] `.streamlit/config.toml` committed
- [x] fonts/favicon committed under `ui/static/`
- [x] backend URL reachable from a secret
- [x] **`API_URL` set to the deployed backend** (Railway)
- [ ] consider the UI-only dependency route (§5)

Backend, wherever it runs:

- [ ] host chosen (any container platform with ≥2 GB RAM and a public HTTPS URL)
- [ ] `GROQ_API_KEY` **and** `OPENROUTER_API_KEY` set
- [ ] Qdrant reachable and `aaa_clinical_v1` populated (`python vectordb/ingest.py`)
- [ ] CORS reviewed if the browser is to call the API directly — it is not today; only
      the Streamlit server process calls it, server-to-server
