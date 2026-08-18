# Reference comparison: `mini-rag` vs this project

Stage-by-stage comparison against the reference implementation kept on disk at
`for ground truth code/` (gitignored, read-only, never imported). Covers the four stages this
project currently has: **chunking, embedding, index/retrieval, evaluation**.

No pipeline code was changed to produce this document. Every claim below cites the file and line
it was read from, and every empirical number was measured from committed artifacts.

---

## 0. What the reference actually is

`for ground truth code/` is **`bakrianoo/mini-rag`** — an educational RAG *service* built for an
Arabic YouTube course (`README.md:1-8` lists 25 tutorial episodes). It is a FastAPI + Celery +
Postgres/pgvector (or Qdrant) deployment scaffold: 3,139 lines of Python, of which the
retrieval-relevant logic is roughly 400.

**This is not the "Project B" already audited in `docs/PROJECT_B_LESSONS.md`.** Project B is
`aaa-clinical-ragnour/` — a different AAA clinical RAG. `mini-rag` is a third reference, not
previously examined, and it is a *generic* RAG scaffold with no clinical or domain content.

That shapes what it can and cannot teach us:

| | `mini-rag` (reference) | This project |
|---|---|---|
| Purpose | deployable multi-tenant RAG service | single-corpus retrieval research + evaluation |
| Strength | service architecture: provider factories, task queues, idempotency, index lifecycle | chunking, provenance, evaluation rigour |
| Chunking | 1 char-threshold splitter, metadata discarded | structure-anchored, token-budgeted, full provenance |
| Evaluation | **none whatsoever** | 3 frozen gold standards, retriever-agnostic scorer |
| Corpus | any uploaded file | 4 fixed AAA guideline PDFs |

**The reference is behind us on three of the four stages.** Its value is concentrated in
*invariants* — how it guarantees that the thing you query is the thing you built — not in
retrieval quality. The proposals in §6 are drawn from those invariants, and I have deliberately
not manufactured recommendations where the reference has nothing to say.

---

## 1. Chunking

### 1a. What the reference does, and why

One function does all of it: `ProcessController.process_simpler_splitter`
(`src/controllers/ProcessController.py:79-105`).

1. **PDF → pages.** LangChain `PyMuPDFLoader` (`:41`) yields one `Document` per page — same
   underlying library we use (PyMuPDF).
2. **Pages are then concatenated and the page boundaries thrown away**:
   `full_text = " ".join(texts)` (`:81`). Nothing downstream can say which page a chunk came from.
3. **Split on newlines**, keeping lines longer than one character (`:84`).
4. **Accumulate lines until a character threshold**, then flush (`:89-97`):
   ```python
   for line in lines:
       current_chunk += line + splitter_tag
       if len(current_chunk) >= chunk_size:
           chunks.append(Document(page_content=current_chunk.strip(), metadata={}))
           current_chunk = ""
   ```
5. **Metadata is set to `{}`** on every chunk (`:94`, `:102`) — the per-page metadata it collected
   at `:61-64` is never attached.

So the reference's **atomic retrieval unit is "a run of consecutive newline-delimited lines whose
combined length reaches `chunk_size` characters."** It is a *typographic* unit, not a semantic one.
Nothing represents a recommendation, a section, or a page.

Default `chunk_size=100` **characters** (~25 tokens) and `overlap_size=20`, both supplied per
request by the API caller (`src/routes/schemes/data.py:6-7`). There is no principled derivation —
they are arbitrary runtime knobs, and 100 characters is far too small to carry a clinical claim.

Provenance is partially recovered *downstream*, not in the chunker: `file_processing.py:214-223`
stamps `chunk_order=i+1`, `chunk_asset_id`, and `chunk_project_id` onto each DB row, and the
`DataChunk` schema reserves a `chunk_metadata` JSONB column (`datachunk.py:17-18`) that the chunker
never populates. Reading order survives as a first-class integer; page and section do not.

**Reference bugs found in this stage** (recorded so they are not mistaken for design):
- `overlap_size` is threaded through four layers (route → task → controller) and **never used**.
  The LangChain splitter that consumed it is commented out at `:66-69`. The reference has *no
  overlap* despite advertising the parameter.
- `if len(current_chunk) >= 0:` (`:99`) is **always true**, so a final chunk is appended even when
  empty. Accidentally, this also means no text is ever dropped.

### 1b. What we do, and where we diverge

Shipped chunker: `notebooks/clinical_atomic_chunking.py` (`DEFAULT_CHUNKER = "atomic"`,
`clinical_chunking.py:845`).

Our **atomic retrieval unit is a structural span**, cut at anchors that describe how a clinical
guideline is typeset (`clinical_atomic_chunking.py:52-59`): `Recommendation N`, NICE-style numbered
IDs (`1.5.4`), and numbered section headings. Page breaks also cut — **except inside a
recommendation**, so pagination can never sever a recommendation (`:176-183`):

```python
prior = [h for h in hard_offsets if h <= off]
if prior and hard_by_off[prior[-1]][0] == REC_ANCHOR:
    continue  # keep the recommendation whole across the page break
```

Page numbers are then *derived* from character offsets of page sentinels (`:127-144`), so
provenance is computed rather than stamped. Spans are token-budgeted against the real tokenizer
(`cc.split_text`), with recommendations given the full 510-token content budget and narrative 220
(`:247`). False-positive anchors (TOC dot-leaders, bibliography entries, author initials) are
rejected by *shape* tests, not by a hardcoded list of this corpus's headings (`:79-102`).

We are ahead of the reference on every dimension of this stage: page provenance preserved rather
than destroyed, semantic boundaries rather than a character threshold, full metadata (19 fields)
rather than `{}`, real overlap (`OVERLAP_TOKENS = 40`, `clinical_chunking.py:47`) rather than an
ignored parameter, and token budgets derived from the model's own `sentence_bert_config.json`
rather than an arbitrary caller-supplied number.

**Divergences worth flagging:**

1. **Two implementations of the same chunker exist, and the shipped one is not the one that was
   evaluated.** This is the most consequential finding in this document; it is developed in §4b
   because it is an evaluation-integrity issue.

2. **Short spans are silently dropped.** `clinical_atomic_chunking.py:244`:
   ```python
   if len(text) < cc.MIN_VALID_CHARS:   # 50 chars
       continue
   ```
   No counter, no warning, nothing in the quality report. `validate_chunks` flags "extremely
   short" chunks that *survive* (`clinical_chunking.py:699`) but never sees the ones dropped
   before they became chunks. A terse recommendation under 50 characters would vanish silently.
   The reference, by accident of its always-true final append, never loses text. This is a real
   asymmetry in accounting, not in intent.

3. **`TARGET_TOKENS = 220` is largely inert.** `split_text` returns text whole when it fits the
   *model limit*, not the *target budget* (`clinical_chunking.py:227-232`):
   ```python
   limit = max_content_tokens(model_name)   # 510
   budget = min(int(target_tokens), limit)  # 220
   if count_tokens(text) <= limit:          # <-- limit, not budget
       return [text]
   ```
   So any span up to 510 tokens is emitted unsplit and `TARGET_TOKENS` never applies. Measured on
   the shipped index: median 203 tokens but **p90 = 468, max = 512** — far above the nominal 220
   "target". This may be intended (the docstring speaks of fitting the window), but then the
   constant is misnamed and reads as a tuned parameter that was never tuned.

4. **No standalone ordinal field.** Reading order exists only inside the `chunk_id` string
   (`f"{document_id}__p{page_start}-{page_end}__c{doc_seq:04d}"`, `:269`), not as an integer field.
   The reference keeps `chunk_order` as a first-class column (`datachunk.py:18`). Ours is
   recoverable by parsing, so this is cosmetic — noted for completeness, not proposed as a change.

5. **Silent metadata fallback.** `:257`:
   ```python
   meta = meta_by_page.get(page_start) or next(iter(meta_by_page.values()))
   ```
   If `page_start` is missing from the page map, the chunk inherits an *arbitrary* page's
   `document_name` / `section_title`. Reachable only via `page_at_offset`'s default of `page = 1`
   (`:139`) for a document whose pages don't start at 1. Low severity, but it fails silently in a
   provenance field.

---

## 2. Embedding

### 2a. What the reference does, and why

Embedding sits behind a provider interface with two implementations, selected by config
(`LLMProviderFactory`). Three decisions matter:

1. **Query and document are different input types.** This is the reference's one genuinely
   valuable retrieval idea. `LLMEnums.py:21-23` defines `DocumentTypeEnum{DOCUMENT, QUERY}`, and
   CoHere maps them to its native `search_document` / `search_query`
   (`LLMEnums.py:17-18`, `CoHereProvider.py:80-82`):
   ```python
   input_type = CoHereEnums.DOCUMENT
   if document_type == DocumentTypeEnum.QUERY:
       input_type = CoHereEnums.QUERY
   ```
   `NLPController` passes `DOCUMENT` when indexing (`:44`) and `QUERY` when searching (`:72`).
   The *intent* is asymmetric encoding: a short question and a long passage are not the same kind
   of text and should not be embedded identically.

2. **Input is truncated before embedding.** `process_text` caps every input at
   `default_input_max_characters = 1000` and is applied to all embedding inputs — including
   queries (`CoHereProvider.py:86`).

3. **No normalization in application code** — cosine is delegated to the vector store's distance
   metric.

**Reference bugs in this stage:**
- The asymmetry **does not work**. `NLPController` passes `DocumentTypeEnum.QUERY.value` — the
  string `"query"` (`:72`) — while `CoHereProvider` compares against the *enum member*
  (`:81`). `"query" == DocumentTypeEnum.QUERY` is `False`, so **queries are always embedded as
  `search_document`.** It also passes the enum member rather than `.value` to the API (`:87`).
- `OpenAIProvider.embed_text` accepts `document_type` and **ignores it entirely** (`:76-98`).
- `process_text` is applied to embedding input in CoHere but **not** in OpenAI (`:89-92`).

So the reference states a good design and then fails to execute it. The design is the transferable
part; the implementation is not.

### 2b. What we do, and where we diverge

Model: `abhinand/MedEmbed-base-v0.1` pinned by commit
`7a90c50263f620dff743eb9794b89a42bfc5d765` (`clinical_chunking.py:31-37`), 768-dim, 512-token
window, resolved from the model's own `sentence_bert_config.json` rather than
`tokenizer.model_max_length` (`:100-107` — a real correctness point, since the two differ by 2× for
MiniLM).

We are ahead on truncation safety: chunks that would overflow **fail the build loudly** rather than
being silently truncated (`clinical_rag.py:54-64`), belt-and-braces with
`embeddable_chunks` and a hard gate in `rebuild_shipped_index.py:32-34`. The reference's crude
1000-character cap silently discards text.

Normalization is correct and consistent: `normalize_embeddings=True` on both the corpus
(`clinical_rag.py:68`) and the query (`:167`), so the bare dot product at `:168` is a true cosine.
Measured row norms on the shipped matrix: min 0.9999999, max 1.0000001. float32 throughout, both
sides. No axis error, no normalized-vs-unnormalized comparison.

**Divergence — the one substantive one in this stage: we encode queries and documents identically.**

Corpus (`clinical_rag.py:66-71`) and query (`:167`) both call plain
`model.encode(..., normalize_embeddings=True)`. There is no prefix, no `prompt_name`, no
`document_type` flag; the model's shipped ST config registers `"prompts": {}`, so nothing is applied
implicitly either. Where the reference *intends* two input types, we have one.

This is **not an oversight** — it is a documented, deliberate choice:
`docs/RETRIEVAL_OPTIMIZATION_EXPERIMENTS.md:423` states "BGE's optional query instruction was
deliberately **not** used, so queries and passages are encoded exactly as the baseline encodes
them," and `:435-436` records that the E5 family was rejected precisely because its mandatory
prefixes would have constituted a prompt change. Holding it constant was the right call for clean
experimental attribution.

The fair criticism is narrower: **it was held constant and never tested.** The relevant facts:

- MedEmbed-base-v0.1 is a fine-tune of `BAAI/bge-base-en-v1.5`; its own card documents no prompt
  template and shows symmetric encoding, so our current behaviour matches its published usage.
- The base model's card *does* document a query instruction —
  `Represent this sentence for searching relevant passages:` — describes it as **optional** for
  v1.5, applied to queries only and never to passages, and specifically **recommends it for
  short-query-to-long-passage retrieval**, closing with "choose the setting that achieves better
  performance on your task."
- Our task is exactly that shape: measured query lengths are **8–13 words** across the frozen sets,
  against passages up to 512 tokens.
- Whether the MedEmbed fine-tune saw the instruction during training is **not documented**, so this
  cannot be settled from model cards. It is an empirical question.

So: defensible as shipped, but an untested hyperparameter in the one regime where the base model's
own guidance says it is most likely to matter.

**Secondary divergence:** the query path has no length guard. The corpus path fails loudly on
overflow; `clinical_rag.py:167` encodes raw query text with no check, so a >512-token query would
be silently truncated. The reference truncates *all* embedding inputs including queries. Currently
theoretical for 8–13 word questions.

---

## 3. Index and retrieval

### 3a. What the reference does, and why

1. **Index identity encodes the embedding configuration.**
   `NLPController.create_collection_name` (`:19`):
   ```python
   return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
   ```
   The vector size is *in the collection name*, so a query issued under a different embedding
   dimension cannot silently land on an index built under another. This is the reference's core
   safety idea: make the index's name a function of the config that produced it.

2. **The vector↔chunk join is explicit and enforced.** The pgvector table carries a `chunk_id`
   column with a real foreign key to `chunks(chunk_id)` (`PGVectorProvider.py:133-134`), and
   `insert_many` refuses to proceed when the arrays disagree (`:240-242`):
   ```python
   if len(vectors) != len(record_ids):
       self.logger.error(f"Invalid data items for collection: {collection_name}")
       return False
   ```
   Nothing is ever matched by position.

3. **Chunking and indexing are chained as one workflow.**
   `process_workflow.py:41-44` runs `chain(process_project_files, push_after_process_task)`, and
   `do_reset` deletes *both* the vector collection and the chunk rows together
   (`file_processing.py:184-192`). You cannot end up with an index built from a different chunk set
   than the one on disk.

4. **Re-run safety by input hash.** `IdempotencyManager.create_args_hash` (`:13-19`) hashes
   `task_name` + all task arguments; `should_execute_task` skips work already completed with those
   exact inputs and re-runs when they change. Work is addressed by its inputs, not by a flag.

5. **Exact search until the corpus is large enough to need approximation** — pgvector builds an
   HNSW index only past `index_threshold = 100` rows (`PGVectorProvider.py:170-171`), searching
   exactly below that. Cosine similarity is `1 - (vector <=> query)` (`:290`).

**Reference bugs in this stage:**
- `default_vector_size: int = 786` — a typo for 768, in both providers
  (`QdrantDBProvider.py:10`, `PGVectorProvider.py:12`).
- The collection *name* uses `vectordb_client.default_vector_size` while the table is *created*
  with `embedding_client.embedding_size` (`NLPController.py:19` vs `:49`). If they disagree, the
  name advertises one dimension and the table holds another — undermining the very invariant §3a.1
  exists to provide.
- `QdrantDBProvider.is_collection_existed` is `async` but is called **without `await`** at `:41`
  and `:51`. A coroutine object is always truthy, so `create_collection` concludes the collection
  already exists and never creates it.
- `PgVectorDistanceMethodEnums.DOT = "vector_l2_ops"` (`VectorDBEnums.py:21`) maps dot product to
  L2 operators.
- `RetrievedDocument` carries only `{text, score}` (`datachunk.py:34-36`), so `chunk_id` and all
  metadata are **discarded at retrieval**. The reference cannot cite a source, and — decisively —
  cannot evaluate by identity. Its prompt template offers only "Document No: N"
  (`locales/en/rag.py:19-24`), with no source attribution.

### 3b. What we do, and where we diverge

`numpy_cosine`: 991 × 768 float32 matrix, exact brute-force search via one matmul
(`clinical_rag.py:168`), no ANN. At this corpus size that is exactly right, and the reference's own
`index_threshold = 100` logic endorses it — exact search below the threshold where approximation
starts to pay.

We retain full metadata on every hit (19 fields, `clinical_rag.py:94-117`) where the reference drops
everything but text and score. That is why we *can* run an ID-level evaluation and it cannot.

**Divergences:**

1. **The vector↔chunk join is positional, not by ID.** `clinical_rag.py:172`:
   ```python
   chunk = dict(index["chunks"][int(idx)])
   ```
   `chunk_id` is not stored alongside the matrix; `embeddings.npy` row *i* is assumed to correspond
   to `embedded_chunks.json` record *i*. The only runtime guard is a length check (`:148-149`).
   Within a single build this is safe by construction (one `valid` list feeds both writes), but a
   **length-preserving** change — a re-sort, a hand-edit, a re-chunk that lands on the same count —
   would pass the check and mis-attribute every citation while scores still looked plausible. In a
   clinical setting that is the worst failure mode available. The reference forbids this
   structurally with a `chunk_id` foreign key and a length assertion at insert.
   **→ Resolved by P3** (§6): the ordered `chunk_id` list is persisted beside the vectors and
   asserted element-wise at load.

2. **`index_meta.json` does not record what it was built from.** It carries model, revision, dim,
   token limit, vector count and timestamp — but **no digest of `chunks.json`**. Nothing in the
   index proves which chunk set produced it. `verify_integrity.py:190-196` compares only *counts*:
   ```python
   len(shipped_chunks["chunks"]) == prof["total_chunks"]
   and shipped_meta["n_vectors"] == prof["indexed_chunks"]
   ```
   Two different chunk sets with matching counts pass this check — which is not hypothetical, as
   §4b shows. The reference addresses this class of problem by hashing all inputs
   (`create_args_hash`) and by encoding config into index identity.
   **→ Resolved by P2** (§6): the index records a content digest of its chunk set, the loader
   refuses to load on mismatch, and `verify_integrity.py` asserts the digest.

3. **The revision pin is bypassed on one query path.** `eval/evaluate.py:188` is
   `SentenceTransformer(index["model_name"])` with **no `revision=`**, while every other consumer
   uses `cc.load_embedder` — including `run_final_evaluation.py:100`, which produced the README
   table. So the published numbers are *not* affected. But the standalone CLI and
   `notebooks/03_aaa_retrieval.ipynb` would silently fetch whatever `main` points at, defeating the
   guarantee asserted at `clinical_chunking.py:33-34`. Currently latent: the pinned SHA is still
   `main`'s HEAD (model untouched since 2024-10-21).
   **→ Resolved by P4** (§6): both call sites now load through `cc.load_embedder`.

4. **The rebuild is coupled — and this is a genuine strength.** `rebuild_shipped_index.py` runs
   chunking then indexing in one script with a hard quality gate between them (`:32-34`: "chunk
   validation FAILED -- not building an index on top of it"). That matches the reference's chained
   workflow and adds a gate the reference lacks.

---

## 4. Evaluation

### 4a. What the reference does

**Nothing.** The reference has no retrieval evaluation of any kind:

- No gold standard, no relevance labels, no P@k / MRR / recall, no test suite, no `tests/`
  directory. A case-insensitive search for `precision|recall|mrr|ndcg|gold_standard|ground_truth|
  evaluat` across all `.py`, `.md` and `.json` files returns **zero hits**.
- `src/utils/metrics.py` is **Prometheus HTTP telemetry** — request counts and latency histograms
  (`:7-8`). Operational monitoring, not retrieval quality.
- Retrieval quality is unmeasurable there by construction: `RetrievedDocument` keeps only
  `{text, score}` (`datachunk.py:34-36`), so there is no identifier to score against.

**Therefore no evaluation recommendation in §6 can be grounded in this reference.** Per the brief,
I am not inventing any. Our evaluation stage is the mature side of this comparison, and the
findings below come from reading our own code — they are reported here because the brief asks for
bugs and arbitrary decisions to be flagged, and listed separately in §7 as *not* reference-derived.

The one thing the reference does say about generation is worth recording as a contrast: its prompt
instructs the model to answer "Based only on the above documents" and to "Ignore the documents that
are not relevant" (`locales/en/rag.py:11,28`) while passing **no source attribution at all** — just
"Document No: N". It has no notion of conflicting sources. Our corpus contains genuinely
conflicting recommendations across four guidelines, and our README is explicit that we surface
rather than reconcile them. The reference offers no help here.

### 4b. What we do, and where we diverge

Our evaluator (`eval/evaluate.py`) is deliberately retriever-agnostic and, in one important respect,
better designed than I expected.

**Relevance is conjunctive** (`:79-103`): a chunk counts as relevant only if **both**
1. its normalised text satisfies at least `min_groups` of the pre-registered `required_facts`
   groups, each group a list of regex alternatives (`:71-76`) — a *content* test; **and**
2. its page range overlaps a pre-registered `answer_passage` in the same document (`:94-98`) — a
   *location* test.

Crucially, the gold standard is keyed on **document + page range**, not chunk IDs. That is why
re-chunking from page-buffer to atomic did not invalidate the labels — a real design win, and the
opposite of the coupling I went looking for.

Metrics are standard and correctly implemented (`:110-164`):
- `p_at(k) = sum(rel_flags[:k]) / k` — textbook precision@k.
- `recall_at(k)` unions the *matched answer-passage indices* over the top k and divides by the
  number of pre-registered passages (`:119-123`) — passage-level recall, correctly deduplicated, so
  one chunk covering two passages counts twice and two chunks covering one passage count once.
- `rr` is the reciprocal of the first relevant rank within the top 10 (`:125`) — i.e. MRR@10,
  reported as "MRR".
- `Relevant_Top1` and `Answering@5` are counts, matching the README's `11/20`, `16/20` form.
- Fully mechanical: **no LLM judge anywhere** in the scoring path.

All three question sets are scored through the same `evaluate_run` code path
(`run_final_evaluation.py:137`), so the sets are comparable to each other. Freeze order is
documented and hash-stamped (`:5-12`, `gold_standard_final20.sha256`).

**Divergences and findings:**

1. **The published `original10` and `heldout18` V1 rows were computed on a chunk set that is not
   the shipped one.** This is the most consequential finding in this document.

   There are **two independent implementations of the V1 atomic chunker**:
   - **shipped** — `notebooks/clinical_atomic_chunking.py`, which produces `data/chunks/chunks.json`
     and `data/embeddings/`;
   - **evaluated** — `eval/experimental_atomic_chunking.py`, which defines its own
     `build_atomic_chunks`, `index_chunks` and `retrieve` (`:289`, `:396`, `:406`) and imports only
     `clinical_chunking`, never the shipped atomic module.

   `run_final_evaluation.py` — the script behind the README table — builds its V1 index from the
   *experimental* module (`:108-110`). Measured from the committed artifacts:

   | | total chunks | indexed | anchors (page/rec/section) |
   |---|---:|---:|---|
   | README V1 row (`final_evaluation_results.json`) | **1764** | **1004** | 408 / 294 / 302 |
   | shipped (`chunks.json` + `index_meta.json`) | **1760** | **991** | 403 / 298 / 290 |

   These are different chunk sets. Cross-checking the retrieved evidence confirms it: of the 318
   distinct `chunk_id`s retrieved by the evaluated V1 configuration in `final_evidence.json`,
   **134 do not exist in the shipped `chunks.json`**, and 14 rows share an ID but carry different
   text.

   The repo is aware of this and partially closed it. `run_corrected_validation.py` re-ran V1 with
   the citation-heading fix on, producing a profile that matches the shipped counts exactly
   (1760 / 991, same anchor census), and I verified the reconciliation holds at ID level: of the
   **164** distinct `chunk_id`s in the corrected run's evidence, **0 are absent** from the shipped
   `chunks.json` and **0 differ in text**. The corrected run's `final20` metrics are byte-identical
   to the README's V1 row (P@1 0.55, MRR 0.6642, R@5 0.6667, R@10 0.7833, 11/20, 16/20).

   So the **`final20` row is properly corroborated on the shipped chunk set.** The gap is that
   `run_corrected_validation.py` covers `final20` **only**. The README's `original10` and
   `heldout18` V1 rows have never been recomputed on the shipped chunker, and still describe the
   1764/1004 set. Given `final20` showed a delta of exactly zero, the risk is low — but it is
   unverified, and it is cheap to close.

   Note also that `verify_integrity.py`'s promotion check passes throughout, because it compares
   counts (§3b.2) — and 1760/991 does match the corrected profile. The check is not wrong; it is
   just not strong enough to have caught the 1764/1004 discrepancy.

   This is precisely the failure mode the reference prevents structurally: **one** chunking
   implementation, invoked by **one** chained workflow, with outputs addressed by an input hash.

2. **`original10` joins query text to gold spec by position.** `queries_for`
   (`run_final_evaluation.py:54-58`) takes `original10`'s question text from
   `clinical_rag.CLINICAL_QUERIES` and pairs it with gold specs by `enumerate(..., start=1)`,
   while `heldout18` and `final20` read both from the same gold record. The `query` field present
   in `gold_standard.json` is ignored for `original10`, and nothing asserts the two agree. I
   verified empirically that they currently match exactly (**0 mismatches across all 10**), so this
   is latent fragility rather than an active bug — but it is the same positional-join pattern the
   reference eliminates with explicit record IDs.

3. **`recall_at` returns 0.0 when a question has no answer passages** (`:123`,
   `if n_passages else 0.0`). A malformed gold entry would silently depress mean recall rather than
   raise. No such entry exists today.

4. **`MRR` is capped at rank 10** (`:125`, `rel_flags[:10]`). Standard practice, but it is MRR@10
   reported under the unqualified name "MRR" in the README.

---

## 5. Summary of the comparison

| Stage | Reference | Us | Verdict |
|---|---|---|---|
| Chunking | char-threshold, metadata discarded, overlap advertised but unused | structure-anchored, token-budgeted, full provenance, real overlap | **we are far ahead**; adopt nothing |
| Embedding | asymmetric query/doc *intent* (broken in code), crude char truncation | symmetric by documented choice, token-exact fail-loud truncation | **we are ahead on safety**; their *intent* is worth testing |
| Index/retrieval | index identity encodes config, explicit `chunk_id` FK, chained rebuild, input-hash idempotency | exact numpy cosine (right at this scale), full metadata retained, join asserted by `chunk_id`, index digest-bound to its chunk set | **their invariants were stronger**; three adopted (P2–P4), the fourth we already had |
| Evaluation | **none** | 3 frozen sets, conjunctive content+page relevance, chunking-agnostic gold, no LLM judge | **nothing to adopt**; findings are our own |

---

## 6. Proposed changes, prioritized

Each proposal names the reference logic it derives from. Nothing here is a free-standing "best
practice" suggestion; items with no reference grounding are in §7 instead and are **not** proposed.

Implementation note for all items: adapt the logic to our conventions (numpy + pandas + committed
JSON artifacts, `pathlib`, type hints, module-level constants). The reference's syntax — SQLAlchemy,
Celery, Pydantic v1, `async`/`await` — does not apply and must not be copied.

### P1 — Make one chunker authoritative, and recompute `original10` / `heldout18` on it
**Priority: high. STATUS: DONE.** **Grounded in:** the reference's single chunking implementation
(`ProcessController.process_simpler_splitter`) invoked by a single chained workflow
(`process_workflow.py:41-44`), so the artifact evaluated is necessarily the artifact shipped; its
interface-plus-backends pattern (`VectorDBInterface` + `VectorDBProviderFactory.create`) in which
variants are *parameters* (`distance_method`, `do_reset`, `index_type`) rather than forked code; and
`process_workflow.py:6` importing `_index_data_content` rather than reimplementing it.
**Our divergence:** §4b.1 — two implementations; the README's `original10` and `heldout18` V1 rows
came from the 1764/1004 experimental set, not the shipped 1760/991 set.

**What was done:**

- `clinical_atomic_chunking.build_chunks` gained two parameters, `keep_page_breaks=True` and
  `reject_citation_headings=True`, both defaulting to the shipped configuration.
  `_heading_is_real`, `find_anchors` and `segment` thread them through.
- `eval/experimental_atomic_chunking.py` lost its duplicate copy — 296 lines of anchor patterns,
  segmenter and chunk-assembly loop removed — and now calls the shipped function with parameters.
  `REJECT_CITATION_HEADINGS` is read at *call* time by thin wrappers, so
  `run_corrected_validation.py:44`'s runtime assignment to it still takes effect.
- Verified behaviour-preserving: the shipped chunker still reproduces `chunks.json` exactly (1,760
  chunks, identical id order, 0 text differences); `ex.build_atomic_chunks(keep_page_breaks=True)`
  now produces that same set (previously 1,764); V2 remains reachable at 1,822; and with the flag
  set to `False` the count returns to **exactly 1,764** — confirming the citation-heading fix was
  the sole cause of the divergence, and that the historical artifact is reproducible from this one
  implementation.
- 29/29 tests pass. `eval/run_p1_shipped_chunker_eval.py` scores all three sets against the shipped
  artifacts, with `final20` reproducing `final_corrected_v1_final20.json` exactly as a control.

**Result:** `original10` and `final20` unchanged in all 8 metrics. `heldout18` moved in three, all
upward — P@3 0.5556→0.5741, P@5 0.4444→0.4556, MRR 0.8148→0.8241 — traceable in full to heldout18
Q7 alone (first relevant hit rank 3→2); the arithmetic closes to rounding. Seven other questions
retrieved renumbered chunk ids from the same pages, which the page-range rule scores identically.
The README now carries both rows, the superseded one retained and labelled.

### P2 — Bind the index to the chunk set it was built from, by content digest
**Priority: high. STATUS: DONE.** **Grounded in:** `IdempotencyManager.create_args_hash`
(`idempotency_manager.py:13-19`), which hashes all inputs to decide whether work must be redone, and
`NLPController.create_collection_name` (`:19`), which encodes the embedding config into the index's
identity so a query cannot hit an index built under a different configuration.
**Our divergence:** §3b.2 — `index_meta.json` records no digest of `chunks.json`, and
`verify_integrity.py:190-196` compares counts only, which is exactly why the 1764/1004 vs 1760/991
discrepancy went unflagged.

**What was done:**

- `clinical_rag.save_index` now stamps three provenance fields into `index_meta.json`:
  `source_chunks_file`, `source_chunks_sha256` (SHA-256 of `data/chunks/chunks.json` as read at
  build time) and `indexed_chunk_ids_sha256` (a digest of the ordered indexed `chunk_id` list). It
  refuses to write an index at all if `chunks.json` is absent, since an index that cannot say what
  it was built from is the defect being fixed.
- `clinical_rag.verify_index_binding` is the single implementation of the check, and
  `clinical_rag.load_index` calls it on every load: a mismatch raises `RuntimeError` with the
  remedy in the message rather than returning an index that will mis-attribute citations.
- `verify_integrity.py` gained a check, *"Shipped index is bound to its chunk set by content digest,
  not by count"*, which calls the loader's own verifier — so the integrity report cannot pass while
  the loader would refuse. The pre-existing count check is kept, relabelled in its own `detail` as a
  profile match that is **not** an identity check.
- `run_stability_checks.py`'s index-integrity check now verifies the binding instead of comparing
  lengths, and `ids.json` was added to its checksum list and to `build_hash_manifest.py`.

The committed index predates these fields, so it had to acquire them without re-encoding: a rebuild
would rewrite `embeddings.npy`, whose SHA-256 is a frozen artifact hash, and float re-encoding is
reproducible only to ~1e-6. `eval/stamp_index_provenance.py` therefore derives the digests from the
artifacts already on disk and preserves every existing `index_meta.json` field including
`created_at_utc`. It stamps nothing until it has **proved** the committed index really is the
shipped selection's output over the committed `chunks.json` — it re-runs
`validate_chunks` → `embeddable_chunks` and requires the result to equal `embedded_chunks.json`
id-for-id and text-for-text. `embeddings.npy` and `embedded_chunks.json` were not written.

`index_meta.json`'s own hash changes as a result, which is unavoidable: it is a build descriptor
carrying a `created_at_utc`, so any rebuild moves it regardless. See §8c for the hash bookkeeping.

### P3 — Carry `chunk_id` with the vector matrix and assert the join
**Priority: medium. STATUS: DONE.** **Grounded in:** the reference's pgvector schema, where every
vector row carries a `chunk_id` with a foreign key to `chunks(chunk_id)`
(`PGVectorProvider.py:133-134`), and `insert_many` aborts when `len(vectors) != len(record_ids)`
(`:240-242`) — the join is explicit and enforced, never positional.
**Our divergence:** §3b.1 — `clinical_rag.py:172` joins by position, guarded only by a length check.

**What was done:**

- `save_index` writes `data/embeddings/ids.json`: the ordered `chunk_id` list, its own digest, and a
  note stating that retrieval joins by position and that this file is what makes the join checkable.
  A JSON sidecar was chosen over `.npz` so the ids stay diffable and `embeddings.npy` stays
  byte-identical to the frozen artifact.
- `verify_index_binding` asserts `ids.json == [c["chunk_id"] for c in embedded_chunks]`
  **element-wise**, and reports the position of the first difference. A length check is retained
  ahead of it for the vectors, since a shorter matrix cannot be compared element-wise at all.
- `experimental_atomic_chunking.load_production_index` — the path every evaluation actually
  retrieves through — had its own `len(chunks) != len(vectors)` check replaced by the same
  verifier. The evaluated path must not be the weaker one; that asymmetry is what §4b.1 was about.
- `tests/test_index_binding.py` (12 tests) proves the guard fires rather than merely existing. Four
  corruptions are applied to a *synthetic* index built in `tmp_path`, so no shipped artifact is
  touched: a length-preserving reorder of `embedded_chunks.json`, a `chunks.json` edited after the
  build, a missing `ids.json`, and an index with the digests stripped. Each must raise. A control
  test loads the uncorrupted synthetic index first, so a broken fixture cannot masquerade as a
  working guard.

### P4 — Pin the model revision on every query path
**Priority: medium. STATUS: DONE.** **Grounded in:** `create_collection_name` (`NLPController.py:19`)
making the index's identity a function of its embedding configuration, so build-time and query-time
config cannot diverge — the invariant an unpinned loader breaks.
**Our divergence:** §3b.3 — `eval/evaluate.py:188` constructs `SentenceTransformer` with no
`revision=`, unlike every other consumer.

**What was done:** both unpinned call sites now load through `cc.load_embedder`, which applies the
revision pin — `eval/evaluate.py`'s `run_retrieval` and the setup cell of
`notebooks/03_aaa_retrieval.ipynb`. The direct `sentence_transformers` import was removed from both,
so the unpinned constructor is no longer reachable from any query path;
`notebooks/clinical_chunking.py:82-83` is now the only place `SentenceTransformer(...)` is
constructed, and that is the function that applies the pin. The published numbers are unaffected:
`run_final_evaluation.py:100` already pinned, which is why this was latent rather than active.
Notebook 03's committed outputs remain from the pre-promotion index (`all-MiniLM-L6-v2`, 1,330
vectors) and were **not** regenerated — that staleness is pre-existing and recorded in §7.

### P5 — Test asymmetric query encoding as a first-class experiment
**Priority: medium (experiment, not a fix).** **Grounded in:** the reference's `DocumentTypeEnum`
(`LLMEnums.py:21-23`) and CoHere's `search_query` / `search_document` split
(`CoHereProvider.py:80-82`) — the design decision that a query and a passage are different input
types. (The reference's own implementation of this is broken — §2a — so only the intent transfers.)
**Our divergence:** §2b — we encode both identically; deliberate and documented, but never A/B
tested, and our 8–13 word queries against ≤512-token passages are exactly the short-query-to-long-
passage regime the base model's card recommends the instruction for.
**Change:** run one experiment applying `Represent this sentence for searching relevant passages:`
to the **query only**, corpus vectors untouched, scored on `heldout18` and `final20` through the
existing evaluator. Ship only if it wins on both; record the result either way, since a negative
result is what currently justifies the shipped choice. This changes no default until it has
evidence.

### P6 — Account for silently dropped spans
**Priority: low.** **Grounded in:** the reference never loses text — its always-true final append
(`ProcessController.py:99`) means every span reaches the index, in contrast to a silent `continue`.
**Our divergence:** §1b.2 — `clinical_atomic_chunking.py:244` drops spans under 50 characters with
no counter, no warning, and no entry in the quality report.
**Change:** count dropped spans (with document, page and character length) and surface the count in
`validate_chunks`' quality dict, so the number appears in `chunks.json` and in the rebuild output.
Do not change the threshold — this is about accounting, not policy.

### P7 — Guard query length on the embedding path
**Priority: low.** **Grounded in:** `process_text` being applied to *every* embedding input,
queries included (`CoHereProvider.py:86`).
**Our divergence:** §2b — the corpus path fails loudly on overflow (`clinical_rag.py:54-64`) while
the query path (`:167`) has no check at all.
**Change:** count tokens on the query before encoding and raise (consistent with the corpus path's
fail-loud posture) rather than truncate silently. Currently theoretical at 8–13 words; it matters
the moment arbitrary input reaches this function.

### P8 — Make `TARGET_TOKENS` honest
**Priority: low.** **Grounded in:** the reference's `chunk_size` genuinely governing output size
(`if len(current_chunk) >= chunk_size`, `ProcessController.py:91`) — its nominal size parameter is
its actual one.
**Our divergence:** §1b.3 — `split_text`'s early return tests against the 510-token model limit
rather than the 220-token budget (`clinical_chunking.py:231`), so `TARGET_TOKENS` is inert for
every span at or under 510 tokens; measured p90 is 468.
**Change:** decide and document which is intended. Either honour the budget (split at 220, which
**changes the shipped index and requires a full re-evaluation** — so not to be done casually), or
rename the constant to reflect that it is a ceiling applied only to oversized spans and state that
the effective target is the model window. Recommendation: rename and document, since changing it
invalidates the frozen results.

### P9 — Pin library versions
**Priority: low.** **Grounded in:** the reference pins every dependency exactly
(`src/requirements.txt`: `fastapi==0.110.2`, `PyMuPDF==1.24.3`, `qdrant-client==1.10.1`, …).
**Our divergence:** `requirements.txt` uses `>=` throughout (`sentence-transformers>=3.0`,
`pymupdf>=1.24`), so the model weights are pinned by commit while the libraries that consume them
are not — a tokenizer or pooling change in a minor release could move results under a frozen
evaluation.
**Change:** pin the versions the frozen results were produced with, recorded alongside the existing
artifact hashes.

---

## 7. Findings NOT grounded in the reference — reported, not proposed

Real observations from reading our code that the reference has nothing to say about. Listed so they
are visible, explicitly **not** part of §6, and awaiting a separate decision.

- **Unstable top-k tie-breaking.** `clinical_rag.py:169` uses `np.argsort(-scores)`, which is
  quicksort — ties resolve by pivot order, not row index. Every *other* ranker in the repo breaks
  ties explicitly (`clinical_rerank.py:49`, `experimental_hybrid.py:83,96,128`). The shipped index
  has 0 duplicate texts and 0 duplicate vectors, so exact float32 ties are unlikely. The reference
  delegates ordering to the database (`ORDER BY score DESC`) and so offers no guidance here.
- **Zero token headroom.** The embedding gate uses `min(512, 512) = 512` rather than the project's
  own `max_content_tokens()` of 510 (`clinical_rag.py:54` vs `clinical_chunking.py:62-64`). Five
  shipped chunks sit at exactly 512 tokens. They fit precisely, so nothing is truncated — but the
  margin is zero, and other callers do honour the reserve.
- **Count mismatch recorded rather than raised.** `clinical_rag.py:72-74` appends to a `failed`
  list and continues; `save_index` writes the index anyway with the message parked in
  `index_meta["failed_embeddings"]`.
- **Stale documentation.** Notebook 02's markdown still names `all-MiniLM-L6-v2`;
  `docs/RETRIEVAL_OPTIMIZATION_EXPERIMENTS.md:610` describes the archived 1330-vector index
  (max 254 tokens) rather than the shipped 991-vector one (max 512);
  `docs/PROJECT_B_LESSONS.md`'s comparison table still lists this project as "page buffer,
  2,116 / 1,330".
- **No device pinned.** No `device=` anywhere, so CUDA is selected when present; build and query
  could differ by ~1e-6, well inside the existing 1e-4 stability tolerance. Absent seeds are not a
  defect — `encode()` runs under `no_grad` in eval mode with no sampling.

---

## 8. Conventions the reference supplied for *how* to make changes

Two conventions, looked up specifically to ground the P1 follow-ups rather than decided by general
practice — plus, in §8c, the bookkeeping that applying §8b to P2–P4 produced.

### 8a. Avoiding duplicate implementations — the reference has a clear pattern

1. **One contract, N genuinely-different backends; never a second copy of the same logic.**
   `VectorDBInterface` is an ABC declaring the contract (`VectorDBInterface.py:5-51`);
   `QdrantDBProvider` and `PGVectorProvider` implement it; `VectorDBProviderFactory.create()`
   selects one from config (`VectorDBProviderFactory.py:12-31`). Same shape for `LLMInterface` →
   `OpenAIProvider` / `CoHereProvider` → `LLMProviderFactory`.
2. **Variants are parameters, not forked code paths.** `distance_method`, `do_reset`,
   `embedding_size`, `batch_size`, `index_threshold`, `chunk_size` — and notably HNSW-vs-IVFFlat as
   `index_type: str = PgVectorIndexTypeEnums.HNSW.value` (`PGVectorProvider.py:158-159`).
3. **Import and call rather than reimplement.** `process_workflow.py:6` imports
   `_index_data_content` from `tasks.data_indexing` and calls it at `:22`. Shared behaviour lives in
   `BaseController`; `VectorDBProviderFactory` even instantiates `BaseController()` purely to reuse
   `get_database_path`.

This is what P1's refactor followed: one implementation, variants as parameters.

### 8b. Correcting published numbers — the reference supplies only a *partial* analogue

What it has:

- **An append-only revision chain.** `alembic/versions/` holds three migrations, each with a
  `revision` id, a `down_revision` naming its predecessor, and a `Create Date`. The third,
  `243ca8b683b0_update_celery_task_executions_table_.py`, changes what the second created — and does
  so by adding a new dated revision that names its predecessor, **not** by editing the earlier file.
  Prior revisions are immutable; current state is the result of applying the chain.
- **Generate, don't transcribe.** The only numbers it publishes are Prometheus metrics, rendered
  live by `generate_latest()` (`metrics.py:36`).

What it does **not** have, stated explicitly: no CHANGELOG, no VERSION file, and zero repo-wide hits
for supersede / deprecate / obsolete / archive / historical / stale / regenerate. **Its README
reports no metrics at all**, so there is no direct analogue for a results table whose published
numbers turned out wrong. The convention is structural, not literal.

This repo already implements that structure, more thoroughly than the reference: 28 immutable
artifacts in `eval/runs/`; `final_corrected_v1_final20.json`'s explicit `what_this_is` /
`overwrites_nothing` fields, with `metrics_historical_v1_fix_off` and `historical_top1_chunk`
keeping superseded values inside the new record; `verify_integrity.py:115,128` asserting no
historical run was overwritten; `data/archive_baseline_index/` preserving the replaced index; and
`build_experiment_history.py` regenerating its outputs *by reading* `eval/runs/`, emitting the
literal `"Not available in preserved artifact."` rather than guessing.

The gap: **nothing generates the README results table.** It is the one place the repo's own
generate-don't-transcribe convention is not applied, which is how the stale `heldout18` row
survived. The P1 correction therefore followed the append-only pattern (superseded row retained and
labelled, pointing at the new artifact) rather than an in-place edit.

### 8c. Artifact hash bookkeeping for P2–P4

P2 changes `index_meta.json` and P4 changes `eval/evaluate.py`, both of which are hash-tracked. What
follows is the record, so a moved hash is never mistaken for tampering. Per §8b the authority is the
*generated* manifest (`eval/final_artifact_hashes.json`, regenerated by `build_hash_manifest.py`),
not a transcription here.

**Changed by this work:** `data/embeddings/index_meta.json` (three provenance fields added),
`eval/evaluate.py` (P4), `notebooks/clinical_rag.py`, `eval/experimental_atomic_chunking.py`,
`eval/verify_integrity.py`, `eval/run_stability_checks.py`, `eval/build_hash_manifest.py`,
`notebooks/03_aaa_retrieval.ipynb`. **Added:** `data/embeddings/ids.json`,
`eval/stamp_index_provenance.py`, `tests/test_index_binding.py`.

**Deliberately unchanged, byte-for-byte:** `data/embeddings/embeddings.npy`,
`data/embeddings/embedded_chunks.json`, `data/chunks/chunks.json`, all three gold standards, and
every file in `eval/runs/`. The frozen evaluation record is untouched; only the index's description
of itself gained fields.

**A pre-existing staleness this exposed.** Two hash records were *already* out of date at `HEAD`,
before any change in this session:

- `eval/final_artifact_hashes.json` disagreed with the working tree on `data/chunks/chunks.json`,
  `data/embeddings/embedded_chunks.json`, `data/embeddings/index_meta.json` and
  `notebooks/clinical_rag.py`. Its own docstring says it must be re-run whenever anything it hashes
  is regenerated; promoting V1 regenerated exactly those files and it was not re-run. Regenerating
  it is the documented remedy and part of this work.
- `notebooks/final_evidence_evaluation.ipynb`'s cell-22 `TRACKED` dict disagreed on four of its seven
  entries (`chunks.json`, `embeddings.npy`, `embedded_chunks.json`, `index_meta.json`) for the same
  reason — the notebook was written before the promotion replaced them. Its committed output shows
  `All tracked artifacts unchanged: True`, which was true when it ran and has not been true since.

That notebook is **deliberately not retro-edited.** Its committed outputs are the handoff evidence,
and §8b's convention is that prior revisions are immutable — editing the expected hashes while
leaving the displayed evidence in place would reproduce the precise pathology §9 documents: a report
asserting a match that no longer holds. The correct reading is that cell 22 describes the artifact
set as of the handoff, not as of now, and `eval/final_artifact_hashes.json` is the current record.

The wider gap this points at is already named in §8b: **nothing generates the README results table,
and nothing regenerates that notebook.** Until they are generated, hash expectations transcribed
inside them will keep drifting from the artifacts they describe.

---

## 9. Known limitation: the `final20` freeze hash no longer matches the file

**Pre-existing, not caused by P1, and deliberately not repaired.** Running
`eval/verify_integrity.py` returns **18 pass / 1 FAIL**. The failing check is:

> `final20 frozen: file hash == hash stamped at freeze time == hash used by the evaluation`

| | SHA-256 |
|---|---|
| `eval/gold_standard_final20.json` as it exists (and as committed) | `5e02dbb84448a2dc…` |
| `eval/gold_standard_final20.sha256` (stamped at freeze) | `36af11b88f7fd908…` |
| `eval/gold_standard_final20.json.sha256` | `36af11b88f7fd908…` |
| `gold_sha256.final20` recorded in `final_evaluation_results.json` | `36af11b88f7fd908…` |

Evidence that this predates this session:

- `git status` shows the gold standard and both stamp files **unmodified**; nothing in this session
  touched them.
- `git show HEAD:eval/gold_standard_final20.json | sha256sum` equals the working tree exactly
  (`5e02dbb8…`), so this is **not** a line-ending or `.gitattributes` artifact — the file has LF
  endings and zero CR bytes.
- The **committed** `eval/integrity_report.json` recorded this same check as `pass`, with
  `"file": "36af11b8…"`. So the gold file was modified *after* the verifier last ran, and the
  modified version is what was committed while the stale passing report was committed alongside it.

Scope and severity:

- **The published metrics are not affected.** `eval/run_p1_shipped_chunker_eval.py` scores `final20`
  against the file as it exists today and reproduces every published number exactly (P@1 0.550,
  MRR 0.6642, R@5 0.6667, R@10 0.7833, 11/20, 16/20). The two gold versions are metric-equivalent.
- **What is broken is the auditability claim**, which is load-bearing: the README states `final20`
  "was authored and hash-frozen *before* the configuration being tested existed." The freeze hash no
  longer matches the file it certifies.
- The pre-modification bytes are **not recoverable** — the repository has a single commit
  (`497256e`), so no earlier version of the file exists in history.
- `eval/runs/final_corrected_v1_final20.json` records **no** `gold_sha256` at all, so it cannot be
  attributed to either version.

### Decision (recorded 2026-08-18)

**Status: `final20` audit trail broken; metrics unaffected. Accepted as a known limitation.**

No recovery and no re-freeze was attempted, and none should be:

- **Recovery is impossible**, not merely expensive — the bytes do not exist anywhere in the
  repository or its history.
- **Re-stamping the current file would be worse than the failure it hides.** A fresh
  `.sha256` would read as "frozen before the configuration existed" while actually being computed
  after it — asserting the exact property that cannot now be evidenced. A freeze record that is
  known-broken is more honest than one that is quietly re-issued.

Consequences, stated so nobody has to rediscover them:

- `eval/integrity_report.json` stands at **18 pass / 1 FAIL** (19 pass / 1 FAIL after the P2 check
  below). The regenerated, failing report is the committed one; the older passing report was
  **not** restored, because it was passing only by describing a file that had already changed.
- The `final20` **numbers** remain usable and are reproducible on demand from the file as it exists.
  The **pre-registration claim** for `final20` now rests on the freeze *order* documented in
  `run_final_evaluation.py:5-12` and on `gold_standard_final20.json`'s own
  `validation_report`, not on a matching hash.
- Any future frozen set must have its stamp verified in the same commit that introduces it. The
  failure here was not the modification; it was that a *report* asserting the hash matched was
  committed next to a file where it did not.

