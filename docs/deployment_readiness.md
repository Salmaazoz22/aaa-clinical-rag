# Deployment Readiness

**Verdict: NOT production ready. It is a reproducible research pipeline, which is what it was
built to be.**

Every claim below is backed by a check in `eval/stability_report.json` that actually ran. Checks
that could not be run are recorded there with status `not_run` and a reason — they are never
reported as passing. Reproduce with:

```bash
python eval/run_stability_checks.py
```

---

## 1. What is verified

| Property | How it was verified | Result |
|---|---|---|
| Deterministic chunking | Rebuilt twice from `data/processed` and compared IDs, text and page ranges | identical — the chunker is deterministic |
| Deterministic chunk IDs | Uniqueness + format `{document_id}__p{start}-{end}__c{seq:04d}` | unique, well-formed |
| Pinned embedding revision | `index_meta.model_revision` vs the revision pinned in code | matches `7a90c502…` |
| Index integrity | vector count vs metadata count; L2 norms | aligned, norms ≈ 1.0 |
| Index reproducibility | re-embedded a 64-chunk sample and compared to stored vectors | max abs delta below tolerance |
| Token safety | recomputed every indexed chunk's token count with the model's own tokenizer | 0 over the limit, 0 stale counts |
| Fail-loud validation | fed `embeddable_chunks` a deliberately oversized chunk | raises `ValueError` as required |
| Malformed input | empty, whitespace-only, single char, 200×-repeated, non-ASCII, punctuation-only, SQL-like, duplicate query | none raised |
| Ranking determinism | same query retrieved 3× | identical rankings and identical scores |
| Latency | timed the 10 clinical queries on CPU | see report |
| Footprint | vectors on disk and in memory | see report |
| Test suite | `pytest tests -q` | see report |
| Artifact checksums | SHA-256 of all gold standards, chunks, vectors, index metadata and pipeline source | recorded |

## 1a. One check FAILS by design of an earlier decision — read this before rebuilding

**`Committed chunks.json still reproduces from the CURRENT code` → FAIL.**

The chunker is deterministic; the *artifact* is stale relative to the code. `chunks.json` holds
2,116 chunks with a 254-token maximum, produced when `DEFAULT_EMBED_MODEL` was
`all-MiniLM-L6-v2` (256-token window). Experiment 7 adopted MedEmbed-base-v0.1 (512-token window)
and deliberately re-embedded the **same** chunks without re-chunking, so that the model swap was
the only variable in that experiment. Since `split_text` budgets against whichever model is
active, `run_chunking()` today produces **2,045** larger chunks instead.

**No evaluation is invalidated** — every run was scored against the committed artifacts, which
are internally consistent (2,116 → 1,330 indexed → 1,330 vectors, 0 over the token limit). But
**do not casually re-run `run_chunking()` against this repository**: it will silently replace the
corpus that every frozen evaluation was measured on.

**Fix before any rebuild:** record `chunker_model` and `chunker_token_limit` inside `chunks.json`,
make the token budget an explicit argument to `run_chunking()`, then either re-chunk under
MedEmbed and re-run all three frozen evaluations, or keep the MiniLM budget deliberately and
document that choice.

## 2. What is NOT verified, and must not be claimed

- **Clean-checkout reproducibility from the source PDFs.** Verifying it requires deleting
  `data/processed`, `data/chunks` and `data/embeddings` and re-running notebooks 01–03 — which
  would destroy the artifacts every preserved evaluation is scored against. What *was* verified
  is chunking determinism from `data/processed` and index reproducibility by re-embedding. **The
  unverified link is PDF → `data/processed`.**
- **Answer quality.** This project evaluates retrieval only. There is no generation step, no
  citation-faithfulness check and no hallucination evaluation.
- **Clinical safety.** Nothing here establishes that surfacing these passages is safe for
  clinical decision-making.

## 3. Gaps that block production

| Gap | Impact | Notes |
|---|---|---|
| **No service** | There is no API, server or CLI entry point for retrieval | retrieval is a library call from a notebook |
| **No logging or observability** | A failure in production would be silent and undiagnosable | `clinical_rag.retrieve` emits nothing — no timings, no request ids |
| **No abstention threshold** | An empty, nonsensical or wholly out-of-scope query still returns 10 chunks with low similarity | there is no floor below which the system declines to answer; **this is the most clinically dangerous gap** |
| **No authentication, rate limiting or monitoring** | — | none exist |
| **Exhaustive NumPy search** | Exact and reproducible at 1,330 vectors; will not scale unchanged | not an ANN service |
| **CPU-only latency** | Dominated by query encoding, not by the dot product | no GPU path exercised |
| **No conflict handling** | The corpus contains genuinely conflicting guideline recommendations (repair threshold in women; preoperative beta blockers; screening in women) | the system surfaces them without reconciling or flagging the conflict |

## 4. What would make it deployable

In dependency order — the first two are the ones that matter clinically:

1. **Add an abstention rule.** Define a similarity floor (and validate it on a set frozen for
   that purpose) below which the system returns "no confident evidence found" rather than ten
   weak chunks. Do not tune this against `final20`; it has now been used.
2. **Surface guideline conflict explicitly.** When retrieved passages from different guidelines
   disagree, the response must say so rather than presenting one as the answer.
3. **Add structured logging** to the retrieval path: query, latency, top-1 score, chunk ids
   returned, index version.
4. **Wrap retrieval in a service** with health checks, and pin the index by checksum at startup
   so a mismatched index fails loudly instead of serving silently wrong results.
5. **Verify clean-checkout reproducibility** in CI, in a scratch copy, from the four PDFs.
6. **Independent clinical review of the gold standards.** All three sets were authored by the
   same agent that ran the experiments.

## 5. Operational notes

- **Model:** `abhinand/MedEmbed-base-v0.1`, revision `7a90c50263f620dff743eb9794b89a42bfc5d765`,
  768-dim, 512-token window, L2-normalised.
- **Index:** `numpy_cosine`, 1,330 vectors, float32, exhaustive.
- **Determinism:** no sampling anywhere in the retrieval path; identical inputs give identical
  rankings and identical scores.
- **Failure mode to watch:** if `data/embeddings/embeddings.npy` and `embedded_chunks.json` fall
  out of sync, `clinical_rag.load_index` raises. Keep that check.
