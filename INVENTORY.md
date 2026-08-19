# INVENTORY — Phase 0

Written before any file was modified. Captured on branch `feat/ui-redesign`, branch point `80df532`.
Every schema below was captured from a **live** call against the running FastAPI service, not from
memory or from source reading.

---

## 1. File map

The Streamlit client is `ui/`. Nothing else in the repository is client code.

| File | Lines | Class | What it does |
|---|---:|---|---|
| `ui/__init__.py` | 2 | UI | Package docstring only. |
| `ui/app.py` | 81 | UI | Entry point. Sidebar radio navigation, one health probe per rerun, dispatches to a view module, renders the footer disclaimer. |
| `ui/api_client.py` | 210 | **MIXED** | The HTTP boundary. `ApiError` + `_request` + one function per endpoint. **The caching policy and the "answers are never cached" rule are behavioural, not visual** — this file must be preserved semantically and edited only where presentation demands it. |
| `ui/theme.py` | 361 | UI | Palette constants, one 14 kB CSS string, `apply()`. Entirely replaced by the new token system. |
| `ui/components.py` | 978 | UI | All render primitives: masthead, badges, evidence panel, citation trace, provenance, tables, stat tiles, pipeline trace. Entirely replaced. |
| `ui/demo_questions.py` | 114 | UI | Eight `DemoQuestion` records. Content is copy; keep the questions, restyle the presentation. |
| `ui/views/ask.py` | 326 | UI | Ask page. |
| `ui/views/evaluation.py` | 297 | UI | Evaluation page. |
| `ui/views/safety.py` | 269 | UI | Safety & Abstention page. |
| `ui/views/architecture.py` | 255 | UI | Architecture page + hand-authored SVG. |
| `ui/views/guidelines.py` | 139 | UI | Guidelines & Sources page. |
| `ui/views/technical.py` | 188 | UI | Technical Details page. |
| `tests/test_ui.py` | 474 | UI | Structural + live end-to-end + CSS-injection regression tests. |

**LOGIC — do not touch under any circumstance:**
`generation/`, `retrieval/`, `vectordb/`, `ingestion/`, `eval/`, `data/`, `api/`,
`tests/test_generation.py`, `tests/test_chunking.py`, `tests/test_vectordb.py`,
`tests/test_index_binding.py`, `tests/test_api.py`.

The one genuinely MIXED file is `ui/api_client.py`. Its *semantics* (which endpoint, which
parameters, how a refusal is distinguished from an error) are contract and stay frozen. Only its
error-message copy is presentation.

---

## 2. Page inventory

### Ask — `ui/views/ask.py`
**Purpose:** ask a clinical guideline question; show the answer with its evidence and validator verdict.

Widgets in render order: masthead → system badges → LLM-unavailable notice → `st.text_area` →
`st.button` "Ask" (primary) / "Clear" → `st.expander` "Retrieval settings for this question"
(`st.slider` top_k, `st.slider` threshold) → `st.expander` "Demo questions" (8 × `st.button` +
caption) → `st.spinner` → answer panel → pipeline trace → `st.tabs`(6): Retrieved evidence
(N × `st.expander`), Evidence → answer trace, Retrieval scores, Validator findings, Provenance,
Raw response (`st.download_button` ×2, `st.code`, `st.json` ×2).

Session state: reads/writes `question_text`, `answer_result`, `asked_question`, `answer_error`.
API: `meta()`, `corpus()` on load; `answer(question, top_k, threshold)` on submit;
`chunk(chunk_id)` lazily per opened evidence card.

### Evaluation — `ui/views/evaluation.py`
Frozen retrieval metrics. Widgets: heading → frozen-results notice → 5 stat tiles → 3 ×
(heading + HTML table + `st.expander` with bar rows) → evidence-statistics tiles → interpretation /
limitations columns → provenance table → integrity `st.expander`.
Session state: none. API: `evaluation()`.

### Safety & Abstention — `ui/views/safety.py`
Widgets: heading → notice → 5 × (example card + `st.button` "Run this" + result tiles +
`st.expander` "What the API actually returned") → principles cards → limits notice.
Session state: `safety_demo_result` (dict keyed by example). API: `answer(question)` per example.

### Architecture — `ui/views/architecture.py`
Widgets: heading → inline SVG diagram in a card → legend badges → 6 design-note cards → API-surface table.
Session state: none. API: `meta()`.

### Guidelines & Sources — `ui/views/guidelines.py`
Widgets: heading → 4 stat tiles → 4 document cards → weighting notice → provenance card.
Session state: none. API: `corpus()`.

### Technical Details — `ui/views/technical.py`
Widgets: heading → 6 live-config tiles → 9 component sections → repo-layout table → provenance panel.
Session state: none. API: `meta()`.

---

## 3. API response schema — the contract the new UI binds to

Captured live. Field names below are exact.

### `GET /health`
```
status: "ok" | "degraded"      qdrant: bool      index: bool
points: int | null             retriever: bool
llm: { configured: bool, provider: str|null, model: str|null }
```
Example: `{"status":"ok","qdrant":true,"index":true,"points":991,"retriever":true,
"llm":{"configured":true,"provider":"groq","model":"openai/gpt-oss-120b"}}`

### `GET /v1/meta`
```
model, revision, dimensions:int, collection, chunk_count:int, index_status, distance, vector_store
connection: { mode, url, local_path, collection, prefer_grpc, timeout_s, exact_search, api_key_supplied }
index_provenance: { source_chunks_file, source_chunks_sha256, indexed_chunk_ids_sha256,
                    n_vectors:int, token_limit:int, max_chunk_tokens:int, built_at_utc,
                    index_meta_sha256 }
generation: { provider, model, api_key_supplied:bool, top_k:int, score_threshold:float, temperature }
```
**`generation.score_threshold` (0.75) is the caliper threshold for the idle state.** Never hardcode it.

### `GET /v1/corpus`
```
documents: [ { document_id, document_name, document_type, source_file, source_organization,
               authors:[str]|null, publication_year:int, source_type, credibility_note,
               public_access:bool|null, source_url, page_count:int, extraction_status,
               extraction_library, indexed_chunks:int } ]
n_documents:int   total_indexed_chunks:int   chunk_counts_source:str
provenance: { file, sha256 }
```
**`publication_year` lives here and nowhere else.** The evidence card's year is obtained by joining
`hit.document_id → corpus document → publication_year`.
**Coverage bar** = `indexed_chunks / total_indexed_chunks`. Real, not derived-by-guess.

### `GET /v1/evaluation`
```
frozen:bool  shipped_config:str  embedding_model  embedding_revision  retrieval  eval_depth:int
gold_sha256: { original10, heldout18, final20 }
metrics: { original10|heldout18|final20: { <config>: { metrics: {n_queries,P@1,P@3,P@5,MRR,
           Recall@5,Recall@10,Relevant_Top1,Answering@5}, per_query:[...] } } }
evidence_statistics: { questions, relevant_in_top_1, relevant_in_top_5, relevant_in_top_10,
                       all_required_fact_groups_covered, no_relevant_evidence_in_top_10:[str] }
final_interpretation:str  limitations:[str]  heldout_selection_rule:str
integrity: { all_checks_passed, summary:{pass,fail}, checks:[{check,status,detail}] }
provenance: { <filename>: sha256 }
```
Configs present on every set: `baseline_production`, `V1_atomic_pagesafe` (**shipped**), `V2_atomic_pure`.

### `GET /v1/chunks/{chunk_id}`
```
chunk_id, chunk_text, document, document_id, section, page:int,
page_start:int, page_end:int, recommendation_id, recommendation_grade, evidence_level
```
This is the only source of **full** passage text — `retrieval.hits[].text_preview` is truncated to
240 characters.

### `POST /v1/answer` — normal answer
```
query:str
settings: { provider, model, base_url, api_key_supplied, top_k, score_threshold,
            temperature, max_output_tokens, timeout_s }
safety:   { blocked:bool, signals:[str], rule:str|null, detail:str|null }
refused:  bool
refusal:  null | { reason:str, gate:str }
answer: { recommendation:str,
          supporting_evidence: [ { claim, chunk_id, excerpt } ],
          citations:           [ { document, section, page, chunk_id, retrieval_score, excerpt } ],
          confidence: "High"|"Medium"|"Low"|"Insufficient Evidence",
          disclaimer: str,
          evidence_conflicts?: [ { topic, positions:[{position, source, chunk_ids:[str]}] } ] }
citations_resolved: [ { chunk_id, resolved:bool, document, document_id, section, page,
                        page_start, page_end, retrieval_score:float, rank:int,
                        recommendation_id, recommendation_grade, evidence_level,
                        source_file, model_excerpt } ]
documents_cited: [str]
validation: { ok:bool, is_refusal:bool, n_errors:int, n_warnings:int, codes:[str],
              findings:[ { code, severity:"error"|"warning", message, location, chunk_id,
                           expected?, actual? } ],
              retrieved_chunk_ids:[str], cited_chunk_ids:[str],
              hallucinated_chunk_ids:[str], uncited_claims:[...] }
retrieval: { n_retrieved:int, n_used:int, n_dropped_below_threshold:int,
             used_chunk_ids:[str], dropped:[...],
             hits:[ { rank:int, chunk_id, similarity_score:float, document_id, document,
                      section, page, page_start, page_end, recommendation_id,
                      recommendation_grade, evidence_level, text_preview } ] }
generation: { completion: null | { provider, model, finish_reason, usage:{prompt_tokens,
                completion_tokens,total_tokens}, latency_s:float, response_chars,
                had_reasoning_stream },
              parse_meta: {...}, disclaimer_normalised:bool }
```

**How chunks are represented:** `retrieval.hits[]`. Score field is **`similarity_score`** (float,
cosine). `document_id` is the short code (`NICE_NG156`); `document` is the full title;
`section` is the section title; `page`/`page_start`/`page_end` are integers.

**How citations link to chunks:** `answer.citations[i].chunk_id` → matched against
`retrieval.used_chunk_ids`. `citations_resolved[i]` is the *retriever's own* record for the same
index, carrying `resolved:bool` plus authoritative metadata. Positional: index `i` in
`citations_resolved` corresponds to index `i` in `answer.citations`.

**What the validator emits per citation:** findings carry `location` stamped as
`citations[<i>]` or `citations[<i>].<field>` (`.document`, `.section`, `.page`,
`.retrieval_score`), so a finding attributes to an exact citation index. Codes observed live:
`excerpt_not_in_chunk`, `citation_metadata_mismatch`, `excerpt_stitched`, `empty_excerpt`,
`duplicate_citation`, `wrong_type`, `hallucinated_citation`.

**Abstention vs normal answer:** `refused:true` and `refusal:{reason,gate}` present.
Two live shapes:
- *Safety* — `refusal.gate:"safety:B1"`, `reason:"patient_specific_request"`,
  `safety.blocked:true`, `safety.signals:["explicit_patient_reference","individual_demographics",
  "individual_directed_ask"]`, `generation.completion: null` (**the model was never called**),
  `answer.citations: []` (deliberately none).
- *Threshold* — `refusal.gate:"threshold"`,
  `reason:"all_scores_below_threshold"`, `safety.blocked:false`, `retrieval.n_used:0`,
  all `similarity_score` below `settings.score_threshold`, `generation.completion: null`,
  `answer.citations` populated with the passages that were **examined and rejected**.

In both, `answer.confidence == "Insufficient Evidence"`.

### Derivations the redesign uses (no invented fields)
| UI element | Derived from |
|---|---|
| Grounding footer `n of m verified` | m = `len(answer.citations)`; a citation counts verified when `citations_resolved[i].resolved` is true **and** no validator finding has `location` starting `citations[i]`. Findings with `severity=="warning"` mark it *caution*, `"error"` marks it *failed*. |
| Verdict label | `Abstained` when `refused`; `Grounded` when all citations verified; `Partially grounded` otherwise. |
| Evidence card year | join `hit.document_id` → `/v1/corpus` `documents[].publication_year`. |
| Evidence card "Supports citation ②" | position of `hit.chunk_id` within `answer.citations[].chunk_id`. |
| Caliper threshold line | `settings.score_threshold` (response) / `meta.generation.score_threshold` (idle). |
| Coverage bar | `documents[].indexed_chunks / total_indexed_chunks`. |
| Stage tracker outcomes | `safety` / `retrieval.n_retrieved` / `retrieval.n_used` / `validation`. |

**Dropped for lack of data:** a per-answer *grounding percentage*. Nothing in the response measures
what fraction of the prose is supported; `uncited_claims` was empty in every live capture and counts
claims, not tokens. A percentage would be invented, so the footer uses the honest
`n of m citations verified` count plus a per-citation segmented bar instead.

**Stage tracker honesty:** the API answers in **one** call and emits no intermediate milestones. The
tracker therefore shows all four stages as `pending` with a single indeterminate running indicator
while in flight, then fills in the four **real** outcomes from the response. No stage is ever
animated as completing on a timer.

---

## 4. Version facts

```
Streamlit  1.61.1      (>= 1.40, so container keys, pills, popover, navigation are all available)
Python     3.12.10
pandas 3.0.5 · altair 6.2.2 · requests 2.34.2 · fastapi 0.141.1 · markdown-it-py 4.2.0
```
No upgrade required.

Confirmed available: `st.pills`, `st.segmented_control`, `st.popover`, `st.navigation`, `st.Page`,
`st.dialog`, `st.container(key=…)` → `.st-key-<key>` scoping.

Theme config keys present in 1.61 (178 in the light/base namespace) — the redesign uses:
`base, primaryColor, backgroundColor, secondaryBackgroundColor, textColor, borderColor,
baseRadius, buttonRadius, showWidgetBorder, showSidebarBorder, font, headingFont, codeFont,
baseFontSize, fontFaces, linkColor, chartCategoricalColors`, plus the **entire `[theme.sidebar]`
namespace**, which lets the dark instrument rail be themed natively rather than fought with CSS.

**Two testids named in the brief do not exist in 1.61** and were corrected against the shipped
bundle: `stAppViewBlockContainer` → **`stMainBlockContainer`**, and `stDecoration` → the gradient
bar is no longer emitted under that testid in this version.

---

## 5. Defect list — root causes

### 2.1 Expander labels render literal `_arrow_right` — **root cause found**
Streamlit renders Material icons as
`<span data-testid="stIconMaterial" translate="no">arrow_right</span>`, styled with
`font-family:"Material Symbols Rounded"; font-feature-settings:"liga"`. **The glyph is a font
ligature; the element's text content is the literal icon name.**

`ui/theme.py` contained:
```css
html, body, [class*="st-"], .stMarkdown, p, li, div, label, input, textarea, button {
  font-family: <sans>;  color: var(--ink);
}
```
`[class*="st-"]` matches Streamlit's emotion-generated class names, including that icon span. The
override replaced the icon font, the ligature could not resolve, and the raw name rendered.
Confirmed against the shipped bundle (`static/static/js/`), which contains both `stIconMaterial`
and the literal string `_arrow_right`.

*Fix:* the blanket attribute selector is deleted outright; the new stylesheet uses inline SVG only
(`ui/icons.py`), suppresses Streamlit's own expander icon, and additionally protects
`[data-testid="stIconMaterial"]`'s font-family for any internal icon not replaced.

### 2.2 Button with invisible label — **same root cause**
The identical rule forced `color: var(--ink)` (near-black) onto every Streamlit button regardless of
its background, while `.stButton > button[kind="primary"]` set a dark background **without** setting
a matching text colour. Dark ink on dark blue ⇒ ~1.3:1.

*Fix:* no global `color` on Streamlit internals; each button variant defined as a complete token set
(bg / text / border / hover / active / focus / disabled) and contrast-checked.

### 2.3 Horizontal overflow — **root cause**
`[data-testid="stHorizontalBlock"]` is a flexbox row and its children are flex items, which default
to `min-width:auto`. A flex item cannot shrink below its content's intrinsic width, so any wide
child — the architecture SVG (`min-width:640px`), a long unbroken `chunk_id`, or a wide table —
forces the row wider than the viewport and the page scrolls sideways, clipping the sidebar.

*Fix:* `min-width:0` on flex/grid children plus `max-width:100%`, and every intentionally wide
element wrapped in its own `overflow-x:auto` container.

### 2.4 Vertical dead space — **root cause**
The old stylesheet targeted the legacy `.block-container` class. In 1.61 that element also carries
`data-testid="stMainBlockContainer"` and its padding is set by an emotion-generated class of equal
specificity `(0,1,0)`, injected into `<head>` and re-injected on rerun — so it won the cascade and
the default top padding survived.

*Fix:* target `[data-testid="stMain"] [data-testid="stMainBlockContainer"]`, specificity `(0,2,0)`,
which wins deterministically. The composer→response void is separately removed by the new layout,
which keeps the evidence rail permanently mounted rather than leaving that column empty.

### Additional defects found during Phase 0
- **`st.markdown` payloads are unrendered in `AppTest`**, so page-level tests cannot catch a
  CSS-injection regression. Already covered by CommonMark-based tests added to `tests/test_ui.py`.
- **`text_preview` is 240 chars**, so evidence text requires the `/v1/chunks/{id}` round trip. The
  new evidence rail must fetch lazily and label truncated text honestly.
- **Model output contains U+2011 non-breaking hyphens**, which is why live responses carry
  `excerpt_not_in_chunk` warnings. This is a real validator signal about the model, not a UI bug —
  the redesign must surface it, not hide it.
