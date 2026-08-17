# Lessons from Project B

> **Project B is a separate project**, located at `aaa-clinical-ragnour/` (renamed from
> `aaa-clinical-rag/` during handoff). It is a reference and
> comparison source only. No Project B file, artifact, output or evaluation result was
> modified, deleted or re-run to produce this document. Every Project B figure below was
> measured read-only from its own committed artifacts, or read directly from its source.

---

## 1. What Project B is

An independent AAA guideline RAG build with a different architecture from this project:

| | Project B (`aaa-clinical-ragnour/`) | This project (Project A) |
|---|---|---|
| Layout | `src/` modules + numbered `scripts/` | notebooks + `notebooks/clinical_*.py` |
| Corpus | 3 PDFs (USPSTF, NICE, ESVS) | 4 PDFs (adds SVS 2018) |
| Chunking | structural anchors, document-level | page buffer, page-scoped |
| Chunks | 530 total, 452 indexed | 2,116 total, 1,330 indexed |
| Embedding | `all-MiniLM-L6-v2`, unpinned, 384-dim | `MedEmbed-base-v0.1`, pinned commit, 768-dim |
| Ranking | cosine **+ intent bonuses + query expansion** | cosine only |
| Evaluation | no gold standard, no P@k/MRR | 2 frozen gold standards, retriever-agnostic evaluator |

The full mechanism-by-mechanism audit (19 mechanisms) is in `eval/project_b_comparison.json`.

---

## 2. What Project B did genuinely better

### 2.1 Chunk boundaries come from document structure, not from pagination

This is the one idea worth taking, and it is a real architectural difference.

Project B builds a single document-level string with page sentinels, finds structural anchors
— `Recommendation N` headings, NICE `1.5.4`-style recommendation IDs, numbered section
headings — and cuts the document at those anchors. A recommendation becomes one chunk and is
never sub-split:

```python
# aaa-clinical-ragnour/src/chunk.py
pieces = [body_text] if (is_table or is_rec) else _split_narrative(body_text)
```

This project cuts on pages. Because a full guideline page already exceeds `TARGET_CHARS`, the
buffer flushes at essentially every page end: **2,109 of 2,116 baseline chunks are
single-page**. The chunker has no representation of a recommendation boundary at all;
recommendations are recovered afterwards as metadata, and that recovery reaches only **35 of
the 57** distinct recommendation IDs available.

### 2.2 Page provenance is derived, not stamped

Project B resolves each chunk's `page_start` / `page_end` from the character offsets of its own
span. This project stamps one page range across every piece a flush emits — the exact side
effect that argued against Experiment 1, where bridging widened the page range on 64 chunks
that came wholly from a single page.

### 2.3 Rejecting contents rows before they become anchors

Project B refuses table-of-contents lines as heading anchors. This project catches contents
material only *after* it has become a chunk, via the dot-leader density test.

---

## 3. What Project B got wrong

### 3.1 It silently truncates a third of its own index

Project B enforces no token budget. `MAX_CHUNK_CHARS = 1800` applies only to narrative
sub-splitting; recommendation and table spans bypass it. Measured from its committed index:

| | Project B |
|---|---:|
| Encoder token window (`all-MiniLM-L6-v2`) | 256 |
| Indexed chunks | 452 |
| Indexed chunks **over** the window | **143 (31.6%)** |
| Largest indexed chunk | **23,079 tokens** |
| Tokens silently discarded at encode time | **83,516** |

For roughly a third of its index, the stored vector does not represent the stored text. This
project sizes in tokens against the model's real tokenizer and refuses to encode an oversized
chunk: max 254 tokens, **0** over the limit.

### 3.2 Its ranking function contains one branch per evaluation question

`src/retrieve.py` defines ten intents. There are ten evaluation questions. Several intent
patterns are keyed to a question's exact wording (`risk factors associated`,
`women regarding.*screening`, `differences between open`). On a match, hand-written rules add
up to **+0.35** or subtract up to **−0.25** from the cosine score, including boosts naming
specific answers by ID and by literal sentence:

```python
if chunk.get("recommendation_id") == "11" and "ultrasound screening" in text_lower:
    adj += 0.22
if "important risk factors for aaa include" in text_lower:
    adj += 0.35
```

`_anchor_indices` goes further and force-inserts named chunks into the candidate pool
regardless of their cosine rank — so recall stops being a property of the retriever.

Measured firing behaviour (`_detect_intents` / `_expand_query` imported directly from Project
B's source and run against both question sets):

| | Project B's own 10 questions | This project's 18 held-out questions |
|---|---:|---:|
| Intent rules fired | **10 / 10** | **1 / 18** |
| Query expansion fired | **9 / 10** | **2 / 18** |

The one held-out question that fires does so **wrongly**. H2 asks *"What imaging follow-up is
recommended after endovascular aneurysm repair?"* and matches the `evar_indications` intent —
whose rules apply a **−0.22 penalty** to any chunk mentioning `post.?oper`, `endoleak`,
`after endovascular.*repair` or `surveillance programme`. That is a description of H2's correct
answer. On this question the transferred machinery would not be merely inert; it would push the
right passage down. *(Read from the rules; not measured, because Project B was not re-run.)*

### 3.3 The checker and the ranker share their hardcoded strings

Project B has no gold standard and no P@k/MRR harness. Quality is asserted by
`scripts/verify_retrieval.py`. Its `GOLD_RANK1` and `GOLD_RANK1_AVOID` patterns are the *same
literal strings* as the bonus and penalty patterns inside `_intent_adjustment`:

| `verify_retrieval.py` says the right answer is… | `retrieve.py` does… |
|---|---|
| `ultrasound screening for the early detection` | `+0.18` on that regex |
| `important risk factors for aaa include` | `+0.35` on that literal |
| avoid `prior to abdominal aortic aneurysm repair` | `−0.18` on that regex |
| avoid `post.?oper\|undergone endovascular\|endoleak\|within 30 d` | `−0.22` on that regex |

**9 of 10** of the checker's gold/avoid patterns contain a literal that also appears inside the
ranker's scoring rules. The ranker passes the checker by construction.

This is the most important finding in the audit, and it needs no trust in any reported metric:
it is visible by opening two files side by side.

### 3.4 What the reported numbers mean in that light

The previously recorded Project B evidence — P@1 0.40 → 0.90 and MRR 0.492 → 0.917 on its own
ten questions, versus ΔP@1 = 0 and ΔMRR ≈ +0.003 on held-out questions — is exactly the
signature the code predicts. A +0.50 P@1 swing where the machinery fires and no swing where it
does not is not a retriever that generalises; it is a set of rules that store the answers.

*(These figures were provided rather than reproduced: Project B contains no metric harness, so
they cannot be regenerated from what is committed there, and it was not re-run to obtain them.)*

---

## 4. The transferred idea, tested independently

The chunking idea was isolated from all of the query machinery and tested against this
project's **unchanged** frozen evaluator, on both question sets.

Implementation: `eval/experimental_atomic_chunking.py`. It writes nothing to `data/`, is
imported by no production code, and every variant is chunked, embedded and scored in memory.
Retrieval is pure dense cosine on pinned MedEmbed — no reranking and no query rules of any
kind. Section titles are deliberately left on the baseline's page-level derivation, because
Experiment 3 established that `section_title` is never embedded.

**Variants**

| | What it is | Purpose |
|---|---|---|
| `control_production` | the shipped index, untouched | baseline |
| `V1_atomic_pagesafe` | anchors cut; page breaks still cut narrative, never a recommendation | the transfer, conservative |
| `V2_atomic_pure` | anchors only | the transfer, closest to Project B |
| `V3_size_control` | **baseline algorithm, no anchors**, budgets enlarged to V1's mean chunk size | separates *structure* from *bigger chunks* |
| `V4_pagespan_control` | production ranking unchanged, page ranges widened to V2's mean span | separates *retrieval* from the frozen rule's *page-overlap* term |

V3 and V4 exist because two things could produce a gain without any retrieval improvement at
all: the evaluator's relevance rule rewards a chunk that satisfies more fact groups (so longer
chunks score better), and it also requires page-range overlap (so wider spans score better).
Neither control changes what is retrieved in the way the real variants do.

### 4.1 Results — and what the controls did to them

Chunk shape (`eval/runs/exp12_atomic_chunking.json`):

| variant | indexed | mean tok | max tok | over limit | pages/chunk | % multi-page |
|---|---:|---:|---:|---:|---:|---:|
| `control_production` | 1,330 | 191.6 | 254 | 0 | 1.005 | 0.5% |
| `V1_atomic_pagesafe` | 1,004 | 221.4 | 512 | 0 | **1.232** | 14.8% |
| `V2_atomic_pure` | 1,081 | 217.1 | 512 | 0 | **5.568** | 81.6% |
| `V3_size_control` | 1,259 | 199.8 | 503 | 0 | 1.003 | 0.3% |
| `V4_pagespan_control` | 1,330 | 191.6 | 254 | 0 | 4.96 | 100% |

P@1 and MRR, as scored, and then with the page term removed:

| | orig-10 P@1 | orig-10 MRR | held-18 P@1 | held-18 MRR | orig-10 P@1 *start-page-only* | held-18 P@1 *start-page-only* |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.500 | 0.619 | 0.556 | 0.697 | 0.500 | 0.556 |
| **V1** | 0.600 | 0.775 | 0.722 | 0.815 | **0.600** | **0.722** |
| V2 | **0.700** | **0.820** | 0.722 | 0.824 | 0.400 | 0.667 |
| V3 size control | 0.500 | 0.621 | 0.556 | 0.723 | 0.500 | 0.556 |
| V4 page-span control | 0.600 | 0.686 | 0.667 | 0.769 | — | — |

**Decomposing the effect.**

- **Effect of chunk size — negligible.** V3 moves P@1 by 0.000 on both sets (MRR +0.0014 /
  +0.0259). It is a *partial* control: it reached a 199.8-token mean against V1's 221.4 (baseline
  191.6), so it covers roughly 28% of the size gap, not all of it. The residual is not ruled out,
  but nothing suggests size is the driver.
- **Effect of page-span overlap — large and real.** V4 changes **nothing** about retrieval: same
  vectors, same ranking, same retrieved chunks. Only the page metadata is widened. That alone
  yields **P@1 +0.10 (original 10)** and **+0.1111 (held-out 18)**. Roughly a tenth of P@1 is
  available for free to any chunker that produces wider page spans.
- **Effect of structure — real, and it survives the strictest test.** Under start-page-only
  scoring the page term is eliminated entirely; the control is unaffected (it is 99.5%
  single-page) so the comparison is exact. V1 keeps **+0.100 P@1 / +0.156 MRR** on the original 10
  and **+0.167 P@1 / +0.118 MRR** on the held-out 18. V1's page span is 1.23 versus the
  baseline's 1.005, so it was barely exposed to the confound in the first place.
- **V2 is confounded.** It has the best headline numbers in the entire project, and the weakest
  evidence for them. With the page term removed its original-10 P@1 collapses from 0.700 to
  **0.400 — below the 0.500 baseline**. With 81.6% multi-page chunks the lower bound is very
  loose, so V2's true value lies somewhere between 0.40 and 0.70 and is **unresolved**.

**Remaining uncertainty.** V3's partial size control; single runs on small sets; the anchor
defect described in section 5 was present in every number above; and — most importantly — both
sets used here had by this point been used to *choose between* the variants, which is why
`eval/gold_standard_final20.json` was authored and frozen before the final comparison.

### 4.2 The clean test: final20

`final20` was authored from source guideline text, validated passage-by-passage against the
actual page text, and hash-frozen **before** the comparison below was run. It is the hardest of
the three sets by available evidence (median 4 relevant chunks per question, versus 7 and 20).

| config | P@1 | P@3 | P@5 | MRR | R@5 | R@10 | Rel@1 | Ans@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline production | 0.400 | 0.3167 | 0.250 | 0.5312 | 0.5417 | 0.6083 | 8/20 | 14/20 |
| **V1 atomic page-safe** | **0.550** | 0.3333 | 0.300 | **0.6642** | **0.6667** | **0.7833** | **11/20** | 16/20 |
| V2 atomic pure | 0.550 | 0.3833 | 0.330 | 0.6919 | 0.6417 | 0.7833 | 11/20 | 17/20 |

With the page term removed (`eval/conservative_rescoring.json`), P@1 across all three sets:

| config | orig-10 scored → strict | held-18 scored → strict | final20 scored → strict | % of retrieved chunks multi-page (final20) |
|---|---|---|---|---:|
| baseline | 0.500 → 0.500 | 0.556 → 0.556 | 0.400 → 0.400 | 0.5% |
| **V1** | 0.600 → **0.600** | 0.722 → **0.722** | 0.550 → **0.550** | 17.5% |
| V2 | 0.700 → **0.400** | 0.722 → 0.667 | 0.550 → 0.550 | 74.0% |

**The transferred chunking idea does improve this project's metrics, and it generalizes.** V1's
P@1 is identical with and without the confound on every set. V2 is never better than V1 under
equal scoring and is far worse on the original 10; it remains rejected.

**Decision: ADOPT WITH CAVEATS** — see `eval/final_recommendation.md` for the five caveats and
the promotion path. Project B's chunking *idea* survived independent testing; none of its
retrieval machinery was transferred, and its implementation was not copied.

---

## 5. Scope and honest limitations of the transfer

- **The anchors only exist in two of the four documents.** ESVS 2024 yields 162 recommendation
  and 100 section anchors; NICE NG156 yields 57 and 14. **USPSTF 2019 and SVS 2018 yield none** —
  USPSTF's headings are unnumbered prose and SVS is a slide deck. Those two documents fall back
  to the baseline's page/token splitting. Any measured gain therefore comes from the ESVS and
  NICE portions of the corpus, and the mechanism should not be described as corpus-wide.
- **Project B covers this gap only with hardcoded per-document title lists**
  (`USPSTF_HEADERS`, `NICE_HEADERS`). Those were **not** transferred: they must be hand-extended
  for every new guideline, whereas this project's chunker needs no code change to accept a
  fifth PDF.
- **A defect was found and fixed during the audit.** The first anchor implementation accepted
  numbered *bibliography* lines as section headings — all 15 of USPSTF's "sections" were
  reference entries such as `3 Svensjö S, Björck M, Gürtelschmid M, Djavani`. This is the same
  false-positive class Experiment 3 documented. Author-list, semicolon, multi-comma and
  internal-sentence shapes are now rejected, removing 15 USPSTF and 13 ESVS false anchors. Both
  the pre-fix and post-fix results are reported, so nothing is hidden by the correction.
- Small evaluation sets (10 + 18 questions). No statistical significance is claimed.
