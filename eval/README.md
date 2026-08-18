# Frozen Retrieval Evaluation Harness

> **These metrics are newly established for the current repository because the previous
> evaluation harness / gold standard was not present.** This is a **FRESH BASELINE**. It is
> *not* a continuation of, and is *not* numerically comparable to, any earlier audit's numbers.

## Files

| File | Role |
|---|---|
| `gold_standard.json` | **FROZEN** relevance labels. Never edit. |
| `evaluate.py` | The single scoring implementation. Scores any ranked result. |
| `runs/*.json` | One saved run per experiment: metrics + top-10 with scores, pages, sections, text. |
| `README.md` | This file — methodology of record. |
| `qdrant_migration_verification.json` | **Not an evaluation.** Infrastructure equivalence between the local index and Qdrant. |
| `qdrant_ingestion_report.json` | **Not an evaluation.** What was uploaded to the vector database, and its validation summary. |
| `qdrant_performance.json` | **Not an evaluation.** Measured retrieval latency and footprint. |

The three `qdrant_*` files record an infrastructure migration (`docs/vector_database.md`).
They compute no metric and score no gold standard; they exist to prove the production vector
database returns exactly what the local index returns. The published numbers come from the
files above them and were not re-run.

## Frozen gold standard

```
SHA-256  0b8a443b69960bc5ac20311f0010926a2f131bbb5531ccf369f321f59ed2e5c1
id       aaa-projectA-fresh-2026-08-17
queries  10 (verbatim from clinical_rag.CLINICAL_QUERIES — wording unchanged)
answer passages  42 pre-registered across the 4 guideline PDFs
```

Verify at any time:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('eval/gold_standard.json').read_bytes()).hexdigest())"
```

If that hash ever changes, every comparison in `docs/RETRIEVAL_OPTIMIZATION_EXPERIMENTS.md`
is void.

## How the labels were built

For each of the 10 queries I read the extracted text of the four source PDFs
(`data/processed/pages.json`, plus `recommendations.json` for graded recommendation anchors)
and pre-registered the passages that genuinely answer the question — **before running any
retrieval, and without looking at retriever output**. Relevance is *not* "whatever the
retriever returned".

Each answer passage records `document_id`, a page range, a section/recommendation identifier,
and a `why` field quoting the clinical content that makes it an answer. Multiple passages per
query are allowed, because the guidelines repeat the same clinical fact (e.g. the 5.5 cm
elective-repair threshold appears in ESVS, SVS and NICE).

**No chunk ID appears anywhere in the gold standard.** Labels are anchored to document + page
+ required clinical facts, so they survive re-chunking — which is exactly what the
optimization experiments change.

### Pre-freeze validation

Before freezing, every passage was checked against the **page text** (not against chunks, to
keep the check chunking-independent): does the passage's own page text satisfy that query's
required facts? **42/42 passed, 0 unrepresentable.** A passage that could never be matched
would have silently capped Recall below 1.0.

## Relevance rule (deterministic)

A retrieved chunk is **relevant** to a query iff **both** hold:

1. **Provenance** — same `document_id` as a pre-registered answer passage, and the chunk's
   page span overlaps that passage's page span:
   `chunk.page_start <= passage.page_end AND chunk.page_end >= passage.page_start`.
2. **Required facts** — the normalised chunk text satisfies at least `min_groups` of that
   query's fact groups. A group is satisfied if **any** of its regex alternatives matches.

Both conditions are needed. Provenance alone would reward a reference list or a page footer
that happens to sit on an answer page; facts alone would reward any chunk that repeats the
words "5.5 cm" out of context.

### Text normalisation (frozen)

Applied to chunk text before every regex test: PDF ligatures (`ﬁ`→`fi`, `ﬂ`→`fl`, …) expanded,
curly quotes → ASCII, en-dash / `≥` / `≤` / `‡` glyphs → space, lowercased, whitespace runs
collapsed to one space. Regexes run with `re.IGNORECASE`.

This matters: the ESVS PDF renders "five" as "ﬁve" and "≥ 55 mm" as "‡ 55 mm".

## Metrics

| Metric | Definition |
|---|---|
| **P@k** | (relevant chunks in top-k) / k |
| **MRR** | 1 / (rank of first relevant chunk in top-10); 0 if none |
| **Recall@k** | (distinct pre-registered **answer passages** covered by a relevant chunk in top-k) / (answer passages for that query) |
| **Relevant Top-1** | count of queries whose rank-1 chunk is relevant |
| **Answering@5** | count of queries with ≥1 relevant chunk in the top-5 |

**Recall is passage-level, not chunk-level.** Chunk counts change when chunking changes, so a
chunk-level recall denominator would move between experiments and make runs incomparable.
Passage-level recall has a fixed denominator (42 passages total) under any chunking.

Aggregates are the unweighted mean over the 10 queries (a plain count for Relevant Top-1 and
Answering@5). Retrieval depth is top-10 for every run; P@1/P@3/P@5 and both recalls are read
off that one ranked list. Ties are left in the retriever's own deterministic order — the
evaluator never re-sorts.

## Running it

```bash
python eval/evaluate.py --label fresh_baseline_dense      # dense retrieval (default)
python eval/evaluate.py --label exp4_rerank --rerank      # dense -> cross-encoder rerank
```

Each run writes `eval/runs/<label>.json` containing the aggregate metrics, the per-query
breakdown, and the full top-10 (chunk_id, score, document, page, section, text) for audit.

## Rules that keep this honest

- The 10 query strings are used verbatim from `clinical_rag.CLINICAL_QUERIES`.
- The gold standard is frozen at the SHA-256 above and is never edited during an experiment.
  A judgement that later looks wrong gets recorded as a **separate sensitivity analysis file**;
  the frozen labels stay as they are.
- The evaluator has no access to chunk IDs when deciding relevance, and no query-specific
  branch exists anywhere in the retrieval code.
- Regex patterns in the gold standard express *clinical content* (thresholds, populations,
  modalities). They are scoring-side only and are never consulted by the retriever.
