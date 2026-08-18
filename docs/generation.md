# The generation layer

> ⚠️ Clinical information **retrieval** prototype with a grounded answer layer on top. Not
> clinically validated, not for clinical use. Every answer carries a disclaimer, and the corpus
> contains genuinely **conflicting** recommendations that the system surfaces rather than
> reconciles.

This document covers `generation/` — the layer that turns retrieved guideline chunks into a
structured, cited answer or an explicit refusal. Retrieval itself is **frozen and untouched**:
this layer calls `vectordb.retriever` and works with what comes back. No reranking, no query
rewriting, no intent detection, no per-question branching.

## The path a question takes

```
question
  │  generation.safety.screen_query          patient-specific? refuse locally, before any API call
  ▼
vectordb.retriever.QdrantRetriever.search    frozen: dense cosine, top-K
  │  generation.pipeline.select_usable_hits  evidence floor on the retriever's own score
  ▼
generation.prompts.build_messages            system prompt + CONTEXT block + answer schema
  │  generation.providers                    Groq | OpenRouter, one OpenAI-compatible interface
  ▼
generation.parsing.parse_answer              locate the JSON object, discard any reasoning block
  │  generation.validator.validate_answer    every citation checked against the chunks actually sent
  ▼
structured, cited, possibly-refusing answer
```

`generation.pipeline.answer_question` is the single entry point. Both the retriever and the
provider are **injectable**, which is why the 75 generation tests need neither a vector store nor
a network call.

## The four gates

An answer has to survive all four. Each one is a place the reference implementation had nothing.

| gate | where | condition | outcome |
|---|---|---|---|
| 0 — safety | `safety.screen_query`, **above** retrieval | question asks for diagnosis/dosing for a named individual | refuse locally, no model call, no patient detail leaves the machine |
| 1 — empty retrieval | `pipeline` | nothing came back at all | refuse, `REFUSAL_NO_CHUNKS` |
| 2 — evidence floor | `pipeline.select_usable_hits` | no hit clears `score_threshold` | refuse, `REFUSAL_BELOW_THRESHOLD`, citing what it examined |
| 3 — model judgement | `schema.is_refusal` on the parsed answer | the model read the evidence and judged it insufficient | recorded as `gate: "model"` |

Gate 0 runs before retrieval and before the model precisely so that patient details are never
sent to a third-party API. It is a conservative **pattern gate, not a classifier**: it catches
unambiguous cases and will miss paraphrases. System-prompt rule F2 is the backstop. Both layers
are needed; neither is sufficient alone.

## The evidence floor — `GENERATION_SCORE_THRESHOLD`

Default **0.75**, defined as `DEFAULT_SCORE_THRESHOLD` in `generation/config.py`.

**This is a starting value, not a calibrated one.** It was chosen as a reasonable default for a
corpus whose frozen retrieval sits at P@1 0.55 / Recall@10 0.78. It is not derived from a
calibration study and is expected to be tuned once the generation evaluation has been run a few
times. Two consequences are known up front and are the intended behaviour of a deliberately
conservative floor:

* **It refuses rather than answers on weak retrievals.** In the frozen evidence
  (`eval/final_evidence.json`) the weakest top-1 among the 20 `final20` questions is Q4
  ("indications for endovascular aneurysm repair") at 0.7467 — below this floor, so Q4 refuses.
  Q4 is also the one question with no relevant chunk anywhere in its top-10
  (`eval/final_evaluation_summary.json`), so refusing it is the correct outcome — reached by a
  threshold rather than by knowing which question it is.
* **It trims weak tail chunks out of the context** even when the question is answerable, so an
  answer is usually grounded in fewer than `top_k` chunks.

The floor is applied to the retriever's own cosine score. Nothing is re-ranked and nothing is
re-scored: the order that comes out is the order that went in.

## Citation validation

`generation/validator.py` checks the model's answer against **exactly the chunks that were sent**,
not against the corpus. Errors (prefix `E_`) mean the answer is not trustworthy; warnings
(prefix `W_`) are recorded but do not fail it.

| code | meaning |
|---|---|
| `E_HALLUCINATED_CITATION` | a `chunk_id` that was never sent |
| `E_HALLUCINATED_EVIDENCE_CHUNK` | a conflict position citing a chunk that was never sent |
| `E_UNCITED_CLAIM` | a `supporting_evidence` bullet with no `chunk_id` |
| `E_ANSWER_WITHOUT_CITATIONS` | prose with no citations at all |
| `E_MISSING_FIELD` · `E_INVALID_CONFIDENCE` · `E_NUMERIC_CONFIDENCE` | schema violations |
| `E_REFUSAL_MESSAGE_MISSING` | a refusal that does not carry the canonical refusal message |
| `W_EXCERPT_NOT_IN_CHUNK` | quoted excerpt not found in the cited chunk (ligature-normalised) |
| `W_CITATION_METADATA_MISMATCH` | the model's `document`/`page` disagrees with the chunk's |
| `W_RETRIEVAL_SCORE_MISMATCH` | the model's reported score disagrees with the retriever's |
| `W_DISCLAIMER_NOT_CANONICAL` | disclaimer altered; it is normalised, and the change is recorded |
| `W_NUMERIC_CERTAINTY_PROSE` | numeric certainty language in prose |

`answer_prose()` deliberately **excludes quoted excerpts**: a fact appearing only inside a quoted
excerpt was not *stated* by the answer, it was quoted from the corpus, and that distinction
matters when checking whether an answer actually said something.

## Conflicting evidence

The corpus disagrees with itself, and the answer layer must not paper over that. The documented
case is the diameter threshold for considering elective repair: NICE NG156 and ESVS 2024 set it at
**55 mm**, while ESVS 2024 also records a NAAASP-derived proposal to raise it to **60 mm** on CTA
that the writing committee explicitly declines to adopt.

`evidence_conflicts` is where both positions go, each with its own `chunk_ids`.
`conflict_position_count()` reports the largest number of distinct positions given for any one
conflict; two or more means the answer presented the disagreement rather than resolving it
silently.

**Known gap.** Retrieval surfaces both positions correctly (verified directly against Qdrant —
both are in the top two hits), but the model's prose sometimes merges them into one sentence
instead of presenting two separately cited positions. That is a generation-quality gap, not a
retrieval or citation-fabrication one. It is question G8 in the generation eval.

## Providers

Both are OpenAI-compatible endpoints, pinned in `generation/providers.py`:

| `GENERATION_PROVIDER` | model | intended use |
|---|---|---|
| `groq` | `openai/gpt-oss-120b` | fast iteration during development |
| `openrouter` | `deepseek/deepseek-r1:free` | final / evaluated answers |

Switching provider is one line in `.env`. No key is ever hardcoded, defaulted to a literal,
logged, or written to disk — `generation/config.py` reads everything from the environment, and
`.env.example` documents the variable **names** only.

Both are reasoning models: their reasoning tokens are billed against the same output budget as the
JSON answer, so `GENERATION_MAX_OUTPUT_TOKENS` defaults to **4000**. A limit tuned for the answer
alone truncates the JSON mid-object. `GENERATION_TEMPERATURE` defaults to **0.0** because a
citation that moves between runs is not auditable — though note that neither model is
bit-deterministic even at temperature 0.

## Configuration

| variable | default | notes |
|---|---|---|
| `GENERATION_PROVIDER` | `groq` | `groq` or `openrouter` |
| `GROQ_API_KEY` / `OPENROUTER_API_KEY` | — | leave the unused one empty |
| `GENERATION_TOP_K` | 5 | how many chunks to **request**; changes no retrieval semantics |
| `GENERATION_SCORE_THRESHOLD` | 0.75 | the evidence floor; see above |
| `GENERATION_TEMPERATURE` | 0.0 | deterministic decoding |
| `GENERATION_MAX_OUTPUT_TOKENS` | 4000 | shared with reasoning tokens |
| `GENERATION_TIMEOUT` | 180 | seconds |
| `GENERATION_MODEL` / `GENERATION_BASE_URL` | unset | unset means the pinned default, so an unset variable can never silently change which model was evaluated |

## Running it

```bash
pip install -e .
cp .env.example .env        # then set one API key; never commit .env

# one question, human-readable
python -m generation.pipeline "What ultrasound surveillance interval is recommended for a 35 mm AAA?"

# the full audit record, including the rendered prompt
python -m generation.pipeline "..." --json --show-prompt

# the generation evaluation (writes eval/generation/)
python eval/generation/run_generation_eval.py
python eval/generation/run_generation_eval.py --ids G8,G14,G17    # a subset

# side-by-side claim/source packet for the manual citation read
python eval/generation/manual_citation_review.py
```

## Evaluation

`eval/generation/generation_eval_set.json` (`aaa-generation-eval-v1`) is **new and separate** from
the frozen retrieval gold standards. It scores ANSWER quality and is not comparable to the
retrieval metrics in `eval/final_evaluation_results.json`. It deliberately includes out-of-scope
and patient-specific questions, which the retrieval gold standards do not contain.

Two thirds of the criteria are mechanical — did it refuse when it should, are the citations real,
are the required facts stated — and `run_generation_eval.py` scores those. The remaining criterion
("is each citation the chunk that actually *contains* the claim") needs a person to read the chunk
text, so the script emits the evidence for that read rather than pretending to automate it; that
is what `manual_citation_review.py` formats.

**Current coverage: 6 of the set's questions** (G1, G8, G10, G13, G14, G17), run incrementally
because of free-tier rate limits. `eval/generation/generation_eval_report.md` is the summary of
that run — 5/6 passing, 0 fabricated citations, the one failure being the G8 conflict-presentation
gap described above. It is **not** a full run of the set.

## What this layer does not do

* it does not chunk, embed, re-rank, rewrite queries, or touch the Qdrant collection;
* it does not resolve conflicts between guidelines;
* it has no calibrated abstention threshold (see the floor discussion above);
* it has no authentication, rate limiting, logging or persistence — there is still no service.
  See `docs/deployment_readiness.md`.
