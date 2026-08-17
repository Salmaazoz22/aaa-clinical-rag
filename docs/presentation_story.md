# The Story

Three phrases are used precisely in this document, and they are not interchangeable:

| phrase | means |
|---|---|
| **we tried it** | the experiment ran and its artifact is preserved |
| **it worked** | it improved metrics on a frozen set it was measured against |
| **it generalized** | it improved metrics on a set frozen *before the change existed* |

Almost everything in this project reached the first category. A handful reached the second. Very
little reached the third — and saying so is the point.

---

## Act I — You cannot improve what you cannot measure

The repository arrived with a working pipeline and **no evaluation harness at all**: no gold
standard, no relevance labels, no P@1, no MRR. Four experiments were already queued. They were
**blocked and stayed blocked**, because inventing a gold standard after seeing retrieval output
would have made every later number meaningless.

So the first artifact built was the evaluator: 10 questions, answer passages pre-registered by
document + page + section, and `eval/evaluate.py` — which knows nothing about the retriever, the
model, or chunk identity. Then the baseline was measured: **P@1 0.500, MRR 0.5625**, and
rebuilding from unmodified code reproduced it with a maximum embedding difference of **0.0**.

Every number that follows is attributable to a change, not to build noise.

## Act II — Four honest failures

**Experiments 1–4 all reverted.** Page-spanning recommendations: MRR fell. Header/footer
cleaning: one metric up, one query down. Section-title detection: MRR fell — and it produced the
most useful negative finding in the project, that **`section_title` is never embedded**, so the
work could not have moved retrieval by construction. The cross-encoder reranker traded precision
for a marginal MRR gain and was kept as an opt-in module, not a default.

Four experiments, zero adoptions. That is what a working evaluation harness looks like.

## Act III — Why MedEmbed?

**Why try a different model at all?** Because MiniLM left Q4, Q5 and Q10 with no relevant chunk
anywhere in the top 10, and three chunking-level experiments had failed to move them.

**BGE-base** (Exp 5) lifted Answering@5 from 6/10 to 10/10 but dropped **P@1 0.50 → 0.40**:
better on average, worse where it matters. **BGE + MS-MARCO cross-encoder** (Exp 6) was supposed
to restore the top-1 and instead refuted the hypothesis outright — MRR fell to 0.5294, below BGE
alone. Stacking two general-domain components compounded the domain mismatch.

**MedEmbed-base-v0.1** (Exp 7) was the first change to clear the bar: MRR 0.5625 → **0.6194**,
Recall@10 0.2667 → **0.4683**, Answering@5 6/10 → 8/10, with P@1 held. It was adopted and pinned
by commit hash so a hub-side update cannot silently change the index underneath a frozen
evaluation.

**Lesson:** domain-matched pre-training beat every architectural trick attempted before it.

## Act IV — Why BM25? Why hybrid?

**Why try BM25?** Clinical questions turn on exact tokens — `5.5 cm`, `EVAR`, `55 mm`. It is
entirely plausible that dense embeddings blur exactly the tokens that matter.

It failed decisively: **P@1 0.10, MRR 0.1922**. Guideline questions and guideline prose share
little surface vocabulary, so BM25 matched boilerplate and headings.

**Why try hybrid anyway?** Because a weak signal can still be complementary. Three fusion
weights were tested — 75/25, 50/50, 25/75 — and performance degraded **monotonically** with the
BM25 weight. No configuration beat MedEmbed on any primary metric.

**Lesson:** three points on a monotonic curve rule out a *direction*, not just a setting. This
cheap negative result is also what justified rejecting Project B's keyword-overlap bonus later,
without having to re-litigate it.

## Act V — Why reranking? Why did full MedCPT fail?

Q4's evidence was diagnosed as sitting **outside the top 10 but inside the top 30** — a
retrieval-depth problem, not a corpus gap. A biomedical cross-encoder was the natural fix.

**Full MedCPT reranking failed** for a specific, measurable reason: applied to every query, it
paid for its rescues by displacing answers dense retrieval already had right. On the original 10
it lost **3** correct top-1 answers (P@1 0.50 → 0.40); on the held-out 18 it displaced **4**.

**Lesson:** judge a reranker on what it *breaks*, not only on what it fixes.

## Act VI — Why Selective Policy A? Why was Phase 7 INCONCLUSIVE?

If reranking helps unconfident queries and harms confident ones, rerank **only** the unconfident
ones. Policy A fires when the score margin between rank 1 and rank 10 falls below a threshold.
On the original 10 it looked excellent: MRR 0.6375, Recall@5 0.4067 — better than both dense-only
and full reranking.

But the threshold was derived from those same 10 queries. That is transductive, and it was
labelled as a hypothesis rather than a result *before* it was tested.

Phase 7 froze the threshold, wrote 18 new questions, labelled them before running anything, and
applied the policy unchanged. The outcome was **INCONCLUSIVE**, and it is reported that way
everywhere:

- it preserved all 10 dense-correct top-1 answers and added one — but that gain is **a single
  question**;
- a **+0.0011** change to the threshold reverses it;
- the Q4 structural pattern that motivated the whole idea occurred in **0 of 18** held-out
  questions.

**Lesson:** a held-out test can come back "not proven", and "not proven" must be reported as
such. Policy A has never been retuned and is still labelled experimental.

## Act VII — Why Project B? What did it teach us?

**Why investigate it?** A separate project reported **P@1 0.40 → 0.90** and **MRR 0.492 → 0.917**
on its own ten questions — while showing roughly **zero** improvement on held-out questions. Both
halves of that are informative: something in it worked, and something in it was fitted.

**What it taught us, positively.** Its chunker cuts the document at *structural anchors* and
keeps each recommendation whole. Ours cut on *pages* — 2,109 of 2,116 chunks were single-page, and
the chunker had no representation of a recommendation boundary at all. That idea was worth taking.

**What it taught us, negatively — and this is the more valuable half.** Its ranking function
contains one branch per evaluation question, with bonuses to **+0.35**, penalties to **−0.25**,
boosts naming specific answers by recommendation ID and by literal sentence, and a step that
force-inserts named chunks into the candidate pool regardless of cosine rank.

The decisive evidence is not a metric. Its checker (`verify_retrieval.py`) and its ranker
(`retrieve.py`) **share their hardcoded strings** — `important risk factors for aaa include` is
simultaneously the "correct answer" and a `+0.35` boost. **9 of 10** of its checker's gold/avoid
patterns reuse a literal from its own ranker. The ranker passes the checker by construction.

Measured by running its own code against both question sets:

| | its own 10 questions | our 18 held-out questions |
|---|---:|---:|
| intent rules fired | **10 / 10** | **1 / 18** |
| query expansion fired | **9 / 10** | **2 / 18** |

The single held-out firing is **harmful**: our H2 asks about imaging follow-up *after* EVAR and
matches its `evar_indications` intent, whose rules apply **−0.22** to text mentioning
`post-oper`, `endoleak`, `after endovascular repair` or `surveillance programme` — a description
of H2's own correct answer.

It also has an engineering defect that rules out its implementation regardless: **143 of its 452
indexed chunks (31.6%) exceed its own 256-token encoder window**, silently discarding **83,516
tokens** at encode time. For a third of its index, the vector does not represent the stored text.

**Why query-specific rules were rejected.** Not on taste — on evidence. A rule that can see which
question is being asked can be fitted to it, and this one demonstrably was. Our retrieval path
therefore contains nothing that can branch on the question.

## Act VIII — Why atomic chunking? Did it actually work?

The transferable idea was isolated from all of that machinery and tested: structural anchors,
atomic recommendations, page provenance from sentinel offsets — under our existing hard token
budget, with pure dense cosine and no query rules of any kind.

Then came the part that mattered. **Two things could raise these metrics without any retrieval
improvement at all**, because the frozen relevance rule rewards chunks that cover more text *and*
requires page-range overlap. So two controls were built:

- **V3 (size control)** — baseline algorithm, *no anchors*, budgets enlarged.
- **V4 (page-span control)** — the production index with *identical ranking and identical
  retrieved chunks*, page ranges simply widened.

**What V3 proved.** Chunk size is not the driver. V3 moves P@1 by **0.000** on both sets.
It is a *partial* control — it reached 199.8 mean tokens against V1's 221.4 (baseline 191.6),
covering about 28% of the gap — and that limitation is stated wherever the result is quoted.

**What V4 proved — and this is the finding that changed the conclusion.** Widening page metadata
*alone*, changing nothing about retrieval, yields **P@1 +0.10** on the original 10 and **+0.1111**
on the held-out 18. About a tenth of P@1 is available for free to any chunker with wider page
spans.

**Did V2 actually improve retrieval?** **No — unresolved.** V2 has the best headline numbers in
the project (original-10 P@1 0.700, MRR 0.820) and the weakest evidence for them. Its chunks span
**5.57 pages on average, 81.6% multi-page**. Remove the page term by scoring each chunk on its
start page only, and its original-10 P@1 collapses **0.700 → 0.400, below the 0.500 baseline**.
V2 was **rejected** — the configuration with the highest metrics was the one whose metrics meant
the least.

**V1 is the one that survives.** Its page span is 1.23 versus the baseline's 1.005, so it was
barely exposed. With the page term removed entirely its gains are **unchanged**: **+0.100 P@1 /
+0.156 MRR** on the original 10 and **+0.167 P@1 / +0.118 MRR** on the held-out 18. It is the
first change in the project's history to raise P@1 at all.

## Act IX — The set we had to build

By this point both existing sets had been used to *choose between* V1, V2, V3 and V4. That
consumes them: **neither is held-out with respect to the chunking decision any more.**

So `eval/gold_standard_final20.json` was authored from source guideline text — 20 questions
spanning numeric thresholds, multi-fact answers, negative recommendations, paraphrases,
multi-document topics and deliberately difficult cases — every answer passage validated against
the actual page text, and the whole file **hash-frozen before the final comparison was run**. By
available evidence it is the hardest of the three sets (median 4 relevant chunks per question,
versus 7 and 20).

**And it generalized.** On final20 — the set no configuration had ever seen — V1 raised
**P@1 0.400 → 0.550**, MRR 0.531 → 0.664, Recall@10 0.608 → 0.783, relevant-top-1 8/20 → 11/20.
With the page term removed by start-page-only scoring, V1's P@1 is **unchanged on all three
sets** (0.600 / 0.722 / 0.550).

That is the only result in this project that reaches the third category. Everything else we tried
either failed, or worked only on the set it was measured against.

V2 also posted 0.550 on final20, but 74% of what it retrieved there was multi-page and its MRR
falls from 0.692 to 0.631 once the confound is removed. It is never better than V1 under equal
scoring, and it is far worse on the original 10. It stays rejected.

## Act X — The last gate

One thing still stood between V1 and being shipped. Every V1 and V2 number in the project had been
produced with a known defect present: the anchor detector was accepting numbered *bibliography*
lines — `3 Svensjö S, Björck M, Gürtelschmid M, Djavani` — as section headings. All 15 of USPSTF's
"sections" were reference entries. The fix had been written and gated off, so the historical
artifacts stayed reproducible, but it had never been scored.

So we turned it on and re-ran V1 against `final20` only, tuning nothing:

| metric | historical V1 (fix off) | corrected V1 (fix on) | Δ |
|---|---:|---:|---:|
| P@1 | 0.5500 | 0.5500 | **+0.0000** |
| MRR | 0.6642 | 0.6642 | **+0.0000** |
| Recall@10 | 0.7833 | 0.7833 | **+0.0000** |

All eight metrics identical. The fix removed 15 bogus USPSTF anchors and 13 bogus ESVS ones;
recommendation anchors were untouched. Seven of twenty questions returned a differently
*identified* top-1 chunk — chunk IDs shift when boundaries move — with unchanged rank and
relevance. The removed anchors had only ever fragmented bibliography text, which the content
classifier already excludes from the index.

**A defect can be real, worth fixing, and still immaterial to the result.** Measuring exactly zero
change is a stronger statement than never having found it. V1 passed the gate and is now the
shipped chunker.

Full decision, with all caveats: `eval/final_recommendation.md` and `docs/experiment_history.md`.

## The final limitation

The honest summary of this project is not a metric. It is this:

> Nearly every idea we tried failed, and the ones that appeared to succeed most often were the
> ones whose success was an artifact of how success was being measured. The evaluation's own
> page-overlap term was worth **+0.10 P@1** for free. We only know that because we built the
> control that could invalidate our best result — and it did.

Everything else follows from that: small sets, no significance claims, a partial size control, an
anchor defect fixed but not yet re-scored, and a test set that has now been used once and cannot
be used to make the next decision.
