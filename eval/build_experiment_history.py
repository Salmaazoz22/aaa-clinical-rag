# -*- coding: utf-8 -*-
"""Assemble eval/experiment_history.json and docs/experiment_history.md.

Metrics are READ FROM the preserved run artifacts in eval/runs/ wherever they
exist. Nothing is retyped from memory and nothing is estimated. Where a metric
was never recorded, the field is the literal string
"Not available in preserved artifact." -- never a guess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
RUNS = EVAL_DIR / "runs"
NA = "Not available in preserved artifact."
KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")


def from_run(filename: str) -> Any:
    p = RUNS / filename
    if not p.exists():
        return NA
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("metrics")
    if isinstance(m, dict) and "P@1" in m:
        return {k: m[k] for k in KEYS}
    return NA


def from_corrected(key: str = "metrics_corrected") -> Any:
    """Metrics from the final corrected validation artifact (its own schema)."""
    p = RUNS / "final_corrected_v1_final20.json"
    if not p.exists():
        return NA
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get(key)
    return {k: m[k] for k in KEYS} if isinstance(m, dict) else NA


def from_exp12(variant: str, dataset: str) -> Any:
    p = RUNS / "exp12_atomic_chunking.json"
    if not p.exists():
        return NA
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d["variants"].get(variant)
    if not v or dataset not in v:
        return NA
    return {k: v[dataset]["metrics"][k] for k in KEYS}


def from_phase7(config: str) -> Any:
    p = RUNS / "phase7_heldout.json"
    if not p.exists():
        return NA
    d = json.loads(p.read_text(encoding="utf-8"))
    c = d.get("configs", {}).get(config)
    return {k: c["metrics"][k] for k in KEYS} if c else NA


def from_final(dataset: str, config: str) -> Any:
    p = EVAL_DIR / "final_evaluation_results.json"
    if not p.exists():
        return NA
    d = json.loads(p.read_text(encoding="utf-8"))
    c = d.get("metrics", {}).get(dataset, {}).get(config)
    return {k: c["metrics"][k] for k in KEYS} if c else NA


def E(n, name, problem, hypothesis, intervention, dataset, metrics, evidence,
      result, failure_mode, decision, lesson, phase=""):
    return {
        "n": n, "name": name, "phase": phase, "problem": problem,
        "hypothesis": hypothesis, "intervention": intervention,
        "dataset": dataset, "metrics": metrics, "evidence": evidence,
        "result": result, "failure_mode": failure_mode,
        "decision": decision, "lesson_learned": lesson,
    }


EXPERIMENTS = [
    E(1, "MiniLM baseline (all-MiniLM-L6-v2)",
      "No evaluation harness existed; no baseline retrieval metrics of any kind.",
      "A frozen gold standard plus a retriever-agnostic evaluator will make every later change attributable.",
      "Built eval/gold_standard.json (10 questions, pre-registered answer passages) and eval/evaluate.py. Retrieval: dense cosine over all-MiniLM-L6-v2, 384-dim, 1,330 indexed chunks.",
      "original10",
      {"original10": from_run("fresh_baseline_dense.json")},
      "eval/runs/fresh_baseline_dense.json; docs/BASELINE_SNAPSHOT.md",
      "Baseline established. Rebuilding chunks and embeddings from unmodified code reproduced it exactly (max embedding delta 0.0).",
      "Q4, Q5 and Q10 return no relevant chunk anywhere in the top 10.",
      "BASELINE",
      "A frozen evaluator authored before any change is what makes every later claim checkable. Without it, Experiments 1-4 were blocked and correctly stayed blocked.",
      "Phase 0"),

    E(2, "Experiment 1 - page-spanning recommendations",
      "2,109 of 2,116 chunks are single-page, so a recommendation cut by a page break is split across two chunks.",
      "Welding a genuinely severed clinical unit back together will improve retrieval of that unit.",
      "Bridged a page boundary only when all of: page ends without terminal punctuation, recommendation language in the last 300 chars, next page resumes mid-sentence, section unchanged. Selected 6 of 245 boundaries.",
      "original10",
      {"original10": from_run("exp1_pagespan.json")},
      "eval/runs/exp1_pagespan.json; docs/RETRIEVAL_OPTIMIZATION_EXPERIMENTS.md",
      "MRR fell 0.5625 -> 0.5611. No metric improved. Q1 regressed (first relevant rank 8 -> 9).",
      "Also widened page_start-page_end on 64 chunks that came wholly from one page, losing page precision for no gain.",
      "REVERT",
      "Fixing a real textual defect is not the same as improving retrieval. The defect was genuine (NICE 1.6.1 is severed mid-sentence) and the fix still did not help.",
      "Phase 1"),

    E(3, "Experiment 2 - header/footer cleaning",
      "Repeated page furniture (running heads, copyright footers) is embedded as if it were content.",
      "Removing repeated furniture will raise the signal-to-noise ratio of each chunk.",
      "Frequency-based detection of repeated lines with a safety test, removing 255 line occurrences / 12,800 characters.",
      "original10",
      {"original10": from_run("exp2_furniture.json")},
      "eval/runs/exp2_furniture.json",
      "P@1, P@5 and MRR unchanged; Recall@10 rose 0.2667 -> 0.2917. 2 queries improved, 1 regressed.",
      "A pure frequency test cannot distinguish a running header from a genuinely repeated clinical statement; the safety test was what kept it honest, and it limited the benefit.",
      "REVERT",
      "A change that moves one secondary metric and regresses a query is not an improvement.",
      "Phase 1"),

    E(4, "Experiment 3 - improved section-title detection",
      "~30% of section titles were artifacts: 133 chunks titled with the ESVS running header, 117 'Table 1-continued', one a citation.",
      "More accurate section titles will improve retrieval.",
      "Preferred a heading found inside the chunk; rejected inherited titles failing a heading shape test (-> None rather than a guess).",
      "original10",
      {"original10": from_run("exp3_sections.json")},
      "eval/runs/exp3_sections.json",
      "Detected titles 451 -> 459 with the artifacts eliminated, but coverage fell 87.0% -> 56.1%. MRR fell 0.5625 -> 0.5611. Nothing improved.",
      "STRUCTURAL: section_title is never embedded. build_embeddings encodes chunk_text alone, so titles can only touch retrieval through classify_chunk_content. Exactly one chunk changed class.",
      "REVERT",
      "Establish whether a field is even on the retrieval path before optimising it. Section titles are a provenance and display asset, and must be judged as one.",
      "Phase 1"),

    E(5, "Experiment 4 - optional dense + cross-encoder reranker",
      "Dense-only retrieval leaves Q4, Q5, Q10 unanswered.",
      "A cross-encoder reranking the dense top-N will recover queries dense retrieval ranks poorly.",
      "New opt-in module notebooks/clinical_rerank.py: dense top-30 -> cross-encoder/ms-marco-MiniLM-L-6-v2 -> top-10. No BM25, no query rules.",
      "original10",
      {"original10 (30 candidates)": from_run("exp4_rerank_c30.json"),
       "original10 (20 candidates)": from_run("exp4_rerank_c20.json")},
      "eval/runs/exp4_rerank_c30.json, exp4_rerank_c20.json",
      "At 30 candidates MRR +0.0111 and Recall@10 +0.0250, but P@3, P@5 and Recall@5 all fell. Swapped one correct top-1 (Q3) for another (Q9). At 20 candidates worse on every priority metric.",
      "Trades precision for a marginal MRR gain; a general-domain reranker does not understand clinical specificity.",
      "KEEP AS OPT-IN, NOT DEFAULT",
      "'It found something new' is not the same as 'it is better'. Count what a change costs as well as what it gains.",
      "Phase 1"),

    E(6, "Experiment 5 - BGE-base-en-v1.5",
      "MiniLM's 256-token window and general-web training may under-serve clinical text.",
      "A stronger general-purpose encoder will retrieve better.",
      "Swapped the embedding model to BAAI/bge-base-en-v1.5 (768-dim). Chunking unchanged.",
      "original10",
      {"original10": from_run("exp5_bge_base_en_v15.json")},
      "eval/runs/exp5_bge_base_en_v15.json",
      "Answering@5 rose 6/10 -> 10/10 and every recall metric improved, but P@1 fell 0.50 -> 0.40 and Relevant Top-1 5 -> 4.",
      "Gains concentrated in ranks 2-10 while the top-1 slot got worse.",
      "NOT DEFAULT (validated optional config)",
      "A model can be better on average and worse where it matters most. Decide which metric is the priority before running the experiment, not after.",
      "Phase 2"),

    E(7, "Experiment 6 - BGE-base + MS-MARCO cross-encoder",
      "BGE-base recovered recall but lost top-1 precision.",
      "Reranking BGE's top-30 will restore top-1 precision while keeping its recall.",
      "BGE-base top-30 -> cross-encoder/ms-marco-MiniLM-L-6-v2 -> top-10.",
      "original10",
      {"original10": from_run("exp6_bge_plus_crossencoder.json")},
      "eval/runs/exp6_bge_plus_crossencoder.json",
      "Hypothesis refuted: MRR fell to 0.5294 and Recall@10 to 0.3750, below BGE alone on both.",
      "The reranker actively demoted correct passages; stacking two general-domain components compounded the domain mismatch.",
      "DO NOT MAKE DEFAULT",
      "Stacking two components that are each individually weak on the domain does not produce a strong one.",
      "Phase 2"),

    E(8, "Experiment 7 - MedEmbed-base-v0.1 (biomedical encoder)",
      "Both general-domain encoders traded top-1 precision for recall.",
      "A biomedically pre-trained encoder will improve recall without sacrificing top-1.",
      "Swapped to abhinand/MedEmbed-base-v0.1, pinned at revision 7a90c50263f620dff743eb9794b89a42bfc5d765. 768-dim, 512-token window. Chunking unchanged.",
      "original10",
      {"original10": from_run("exp7_medembed_base_v01.json")},
      "eval/runs/exp7_medembed_base_v01.json, phase1_production_medembed.json",
      "First experiment to clear the bar: MRR 0.5625 -> 0.6194, Recall@5 0.2217 -> 0.3567, Recall@10 0.2667 -> 0.4683, Answering@5 6/10 -> 8/10, with P@1 and Relevant Top-1 held at 0.50 / 5.",
      "None. Q4 remained unanswered in the top 10.",
      "ADOPTED - PRODUCTION DEFAULT",
      "Domain-matched pre-training beat every architectural trick tried before it. Change the representation before adding machinery on top of a weak one.",
      "Phase 2"),

    E(9, "Q4 structural analysis",
      "Q4 (indications for EVAR) returned no relevant chunk in the top 10 under every configuration tried.",
      "Q4's evidence is present in the corpus but ranked outside the evaluation depth, rather than missing.",
      "Inspected the dense top-30 for Q4 and located where relevant chunks actually rank.",
      "original10",
      {"original10": NA},
      "docs/RETRIEVAL_OPTIMIZATION_EXPERIMENTS.md; eval/runs/exp10_medembed_medcpt_rerank.json",
      "Relevant evidence for Q4 sits outside the top 10 but inside the top 30 - a retrieval-depth problem, not a corpus gap.",
      "Not a failure; a diagnosis. It motivated the reranking work in Phases 5-7.",
      "DIAGNOSTIC - no change made",
      "Distinguish 'the corpus does not contain the answer' from 'the retriever did not rank it'. These need completely different fixes.",
      "Phase 3"),

    E(10, "Experiment 8 - BM25 keyword retrieval",
      "Dense retrieval may miss exact clinical terms and numeric thresholds.",
      "Lexical matching will retrieve exact terms ('5.5 cm', 'EVAR') that dense embeddings blur.",
      "Pure BM25 over the same 1,330 indexed chunks.",
      "original10",
      {"original10": from_run("exp8_bm25_keyword.json")},
      "eval/runs/exp8_bm25_keyword.json",
      "Decisively worse: P@1 0.10, MRR 0.1922, Recall@10 0.1983.",
      "Guideline questions and guideline text share little surface vocabulary; BM25 matched boilerplate and headings.",
      "REJECT",
      "A negative result with this margin is valuable: it closed off the entire lexical direction cheaply and justified rejecting Project B's keyword-overlap bonus later.",
      "Phase 4"),

    E(11, "Experiment 9A - hybrid dense 75 / BM25 25",
      "BM25 alone failed, but a small lexical signal might still complement dense retrieval.",
      "A dense-dominant hybrid will add lexical precision without BM25's noise.",
      "Score fusion, 75% dense + 25% BM25.",
      "original10",
      {"original10": from_run("exp9_hybrid_a_75_25.json")},
      "eval/runs/exp9_hybrid_a_75_25.json",
      "MRR 0.6125 vs MedEmbed's 0.6194; Recall@10 0.4850 vs 0.4683. No primary metric beaten.",
      "The lexical channel adds noise roughly as fast as signal.",
      "REJECT",
      "Weighting a bad signal down does not turn it into a good one.",
      "Phase 4"),

    E(12, "Experiment 9B - hybrid dense 50 / BM25 50",
      "As above, at equal weight.",
      "An even blend will balance semantic and lexical matching.",
      "Score fusion, 50/50.",
      "original10",
      {"original10": from_run("exp9_hybrid_b_50_50.json")},
      "eval/runs/exp9_hybrid_b_50_50.json",
      "MRR 0.5617, Recall@10 0.4000 - worse than dense alone.",
      "Degradation scales with the BM25 weight.",
      "REJECT",
      "The monotonic degradation across 9A/9B/9C is itself the evidence: this is a direction, not a tuning problem.",
      "Phase 4"),

    E(13, "Experiment 9C - hybrid dense 25 / BM25 75",
      "As above, lexical-dominant.",
      "Confirms the direction of the trend.",
      "Score fusion, 25/75.",
      "original10",
      {"original10": from_run("exp9_hybrid_c_25_75.json")},
      "eval/runs/exp9_hybrid_c_25_75.json",
      "P@1 0.30, MRR 0.4167 - worst of the three.",
      "As expected from Experiment 8.",
      "REJECT",
      "Three points on a monotonic curve are worth more than one; they rule out the direction rather than one setting.",
      "Phase 4"),

    E(14, "Experiment 10 - full MedCPT cross-encoder reranking",
      "Q4's evidence sits in the top 30 but outside the top 10.",
      "A biomedical cross-encoder (unlike the general MS-MARCO one) will promote it without damaging the rest.",
      "MedEmbed top-30 -> ncbi/MedCPT-Cross-Encoder (revision 71caf65d...) -> top-10, applied to EVERY query.",
      "original10 and heldout18",
      {"original10": from_run("exp10_medembed_medcpt_rerank.json"),
       "heldout18": from_phase7("B_full_medcpt")},
      "eval/runs/exp10_medembed_medcpt_rerank.json; eval/runs/phase7_heldout.json",
      "original10: P@1 fell 0.50 -> 0.40, MRR 0.6194 -> 0.5575, and 3 correct top-1 answers were lost. heldout18: P@3 and MRR improved but 4 correct top-1 answers were displaced.",
      "Reranking every query pays for its rescues by displacing answers dense retrieval already had right.",
      "REJECT",
      "A reranker must be judged on what it BREAKS, not only on what it fixes. Track preserved/lost/gained top-1, not just the mean.",
      "Phase 5"),

    E(15, "Experiment 11 - selective reranking, Policies A / B / C",
      "Full reranking helps low-confidence queries and harms high-confidence ones.",
      "Reranking only when the dense retriever is unconfident will keep the rescues and avoid the damage.",
      "Confidence signal margin_1_10 (score gap between rank 1 and rank 10); rerank only when it falls below a threshold. Policy A threshold 0.035120 = P25 of margin_1_10 over the ORIGINAL 10 queries.",
      "original10",
      {"original10 Policy A": from_run("exp11_selective_rerank_policy_A.json"),
       "original10 Policy B": from_run("exp11_selective_rerank_policy_B.json"),
       "original10 Policy C": from_run("exp11_selective_rerank_policy_C.json")},
      "eval/runs/exp11_selective_rerank_policy_A.json (and _B, _C)",
      "Policy A: MRR 0.6375, P@5 0.4600, Recall@5 0.4067, Recall@10 0.5133 - better than both dense-only and full reranking on the original 10.",
      "TRANSDUCTIVE: the threshold was derived from the same 10 queries it was then scored on. That establishes nothing about generalisation.",
      "PROMISING - REQUIRES HELD-OUT VALIDATION",
      "A threshold fitted on the evaluation set is not a result; it is a hypothesis. Say so before measuring it, not after.",
      "Phase 6"),

    E(16, "Phase 7 - held-out validation of Selective Policy A",
      "Policy A's threshold was fitted on the original 10 questions.",
      "If the policy generalises, the FROZEN threshold will reproduce its benefit on questions it has never seen.",
      "Wrote and froze eval/gold_standard_heldout.json (18 questions, labelled before any retrieval). Applied threshold 0.035120 unchanged - not recomputed, no percentile retaken, no tuning.",
      "heldout18",
      {"heldout18 dense": from_phase7("A_medembed_dense"),
       "heldout18 full MedCPT": from_phase7("B_full_medcpt"),
       "heldout18 Policy A": from_phase7("C_selective_policy_a")},
      "eval/runs/phase7_heldout.json",
      "Policy A preserved all 10 dense-correct top-1 answers and added one (P@1 0.5556 -> 0.6111). But the gain is a single question, a ~3% threshold perturbation reverses it, and the Q4 rescue pattern appeared in 0 of 18 held-out questions.",
      "The mechanism that motivated the policy did not occur even once in the held-out set.",
      "INCONCLUSIVE - EXPERIMENTAL, NOT PRODUCTION",
      "A held-out test can come back 'not proven' rather than yes or no, and 'not proven' must be reported as such. This is the single most important precedent in the project.",
      "Phase 7"),

    E(17, "Project B audit (external reference implementation)",
      "A separate project (aaa-clinical-rag/) reported P@1 0.90 / MRR 0.917 on its own 10 questions but ~zero improvement on held-out questions.",
      "Its chunking architecture may contain a transferable idea, while its retrieval machinery is overfit and must not be copied.",
      "Read-only audit of 19 mechanisms; measured its committed artifacts; imported its _detect_intents/_expand_query and ran them against both question sets.",
      "Project B artifacts and source (no dataset scored)",
      {"n/a": NA},
      "eval/project_b_comparison.json; docs/PROJECT_B_LESSONS.md",
      "Intent rules fire 10/10 on its own questions and 1/18 on our held-out set; query expansion 9/10 vs 2/18. 9 of 10 of its checker's gold/avoid patterns are literal strings copied from its own ranker's scoring rules. 143/452 (31.6%) of its indexed chunks exceed its 256-token encoder window, discarding 83,516 tokens at encode time.",
      "Its checker and its ranker were authored together, so the ranker passes by construction. Its 'evidence' cannot support its claims.",
      "REJECT the retrieval machinery; TEST the chunking idea",
      "Overfitting is often visible in the source before it is visible in the metrics. Read the code, not just the numbers.",
      "Phase 8"),

    E(18, "Experiment 12 V1 - atomic chunking, page-safe",
      "Chunk boundaries are set by pagination, so no chunk corresponds to a clinical unit.",
      "Cutting at structural anchors and keeping recommendations whole will retrieve complete recommendations.",
      "Document-level text with page sentinels; anchors = 'Recommendation N', numbered recommendation IDs, numbered section headings. Page breaks still cut narrative but never a recommendation. Token budget enforced throughout. Pure dense cosine; no query rules of any kind.",
      "original10 and heldout18",
      {"original10": from_exp12("V1_atomic_pagesafe", "original_10"),
       "heldout18": from_exp12("V1_atomic_pagesafe", "heldout_18"),
       "original10 (start-page-only scoring)": from_exp12("V1_atomic_pagesafe", "original_10_startpage_only"),
       "heldout18 (start-page-only scoring)": from_exp12("V1_atomic_pagesafe", "heldout_18_startpage_only"),
       "final20": from_final("final20", "V1_atomic_pagesafe")},
      "eval/runs/exp12_atomic_chunking.json; eval/final_evaluation_results.json",
      "First change in the project's history to raise P@1: +0.10 on original10 and +0.167 on heldout18, with MRR +0.156 / +0.118. Gains are UNCHANGED under start-page-only scoring, which removes the page-overlap term entirely.",
      "Anchors exist in only 2 of 4 documents (ESVS, NICE). USPSTF and SVS yield none and fall back to baseline splitting.",
      "See final decision in docs/experiment_history.md",
      "Structure beat every model-level and reranking trick attempted before it. The unit of retrieval matters more than the machinery ranking those units.",
      "Phase 9"),

    E(19, "Experiment 12 V2 - atomic chunking, pure",
      "As V1, but closest to Project B's shape.",
      "Removing page breaks entirely will maximise the benefit of atomic chunking.",
      "Anchors only; page boundaries never cut. Mean 5.57 pages per chunk, 81.6% multi-page.",
      "original10 and heldout18",
      {"original10": from_exp12("V2_atomic_pure", "original_10"),
       "heldout18": from_exp12("V2_atomic_pure", "heldout_18"),
       "original10 (start-page-only scoring)": from_exp12("V2_atomic_pure", "original_10_startpage_only"),
       "heldout18 (start-page-only scoring)": from_exp12("V2_atomic_pure", "heldout_18_startpage_only"),
       "final20": from_final("final20", "V2_atomic_pure")},
      "eval/runs/exp12_atomic_chunking.json; eval/final_evaluation_results.json",
      "Highest headline numbers of any configuration (original10 P@1 0.70, MRR 0.82). But under start-page-only scoring original10 P@1 collapses 0.70 -> 0.40, BELOW the 0.50 baseline.",
      "CONFOUNDED. Page span 5.57 vs baseline 1.005; the frozen relevance rule requires page-range overlap, so wider chunks are easier to score relevant regardless of retrieval quality.",
      "REJECT - highest metrics, weakest evidence",
      "The configuration with the best-looking numbers was the one whose numbers meant the least. Controls, not leaderboards.",
      "Phase 9"),

    E(20, "Experiment 12 V3 - size-matched control",
      "V1/V2 chunks are larger than baseline, and the relevance rule rewards chunks covering more text.",
      "If the gain is really from chunk SIZE, the baseline algorithm enlarged to the same size will show it.",
      "Baseline page-buffer algorithm, NO structural anchors, token and character budgets enlarged toward V1's mean (target 221 tokens).",
      "original10 and heldout18",
      {"original10": from_exp12("V3_size_control", "original_10"),
       "heldout18": from_exp12("V3_size_control", "heldout_18")},
      "eval/runs/exp12_atomic_chunking.json",
      "Essentially null: original10 P@1 unchanged at 0.50 (MRR +0.0014); heldout18 P@1 unchanged at 0.5556 (MRR +0.0259, Recall@10 -0.0371).",
      "PARTIAL CONTROL. It reached only 199.8 mean tokens against V1's 221.4 (baseline 191.6), because the page buffer flushes at page ends before the enlarged budget binds. It therefore covers about 28% of the size gap, not all of it.",
      "CONTROL - confirms size is not the driver",
      "Report what a control did NOT cover. A control described as complete when it is partial is worse than no control.",
      "Phase 9"),

    E(21, "Experiment 12 V4 - page-span control",
      "The frozen relevance rule requires the chunk's page range to overlap the answer passage's page range, so wider page spans are easier to score relevant.",
      "If page-span width alone can move the metrics, some of V2's advantage is measurement artifact rather than retrieval quality.",
      "Took the PRODUCTION index unchanged - same vectors, same ranking, same retrieved chunks - and widened every chunk's page range by +/-2 pages to match V2's mean span. Nothing about retrieval changed.",
      "original10 and heldout18",
      {"original10": from_exp12("V4_pagespan_control", "original_10"),
       "heldout18": from_exp12("V4_pagespan_control", "heldout_18")},
      "eval/runs/exp12_atomic_chunking.json",
      "Widening page metadata ALONE, with identical retrieval, produced P@1 +0.10 (original10) and +0.1111 (heldout18), MRR +0.0667 / +0.0722.",
      "None - this control worked exactly as intended, and it is the reason V2 was rejected.",
      "CONTROL - quantifies the confound at roughly +0.10 P@1",
      "Build the control that can invalidate your best result. This one cost one function and changed the conclusion.",
      "Phase 9"),

    E(22, "final20 - pre-registered clean test set",
      "By the end of Experiment 12, BOTH existing sets had been used to evaluate the chunking variants, so neither was untouched with respect to that decision.",
      "A third set, authored and frozen before any retrieval, will show whether the chunking gain generalises or was selection.",
      "Authored 20 questions from the extracted recommendation inventory of all 4 guidelines - numeric/threshold, multi-fact, negative-recommendation, paraphrase, multi-document and deliberately difficult types. Every answer passage validated against SOURCE page text. Frozen with SHA-256 before eval/run_final_evaluation.py was executed.",
      "final20",
      {"final20 baseline": from_final("final20", "baseline_production"),
       "final20 V1": from_final("final20", "V1_atomic_pagesafe"),
       "final20 V2": from_final("final20", "V2_atomic_pure")},
      "eval/gold_standard_final20.json; eval/gold_standard_final20.sha256; eval/final_evaluation_results.json; eval/final_evidence.json",
      "See eval/final_evaluation_results.json.",
      "n/a",
      "See final decision in docs/experiment_history.md",
      "Once a held-out set has been used to choose between configurations it is no longer held out. Selection consumes a test set; budget for that in advance.",
      "Phase 10"),

    E(23, "FINAL CORRECTED VALIDATION - V1 with the citation-heading fix ON",
      "Every historical V1/V2 number was produced with a known anchor defect present: numbered bibliography lines ('3 Svensjo S, Bjorck M, Gurtelschmid M, Djavani') were accepted as section headings. The fix was implemented but gated off so the historical artifacts stayed reproducible.",
      "If the defect was immaterial, turning the fix on leaves final20 unchanged. If it was material, the V1 decision must be reopened.",
      "Set REJECT_CITATION_HEADINGS = True, rebuilt V1 chunks, re-embedded, re-scored against final20 ONLY. Nothing else changed: same model, same pinned revision, same dense cosine, same top-k, same gold, same relevance rule. Nothing was tuned -- a gate, not a sweep.",
      "final20",
      {"final20 corrected V1 (fix ON, SHIPPED)": from_corrected(),
       "final20 historical V1 (fix OFF)": from_final("final20", "V1_atomic_pagesafe")},
      "eval/runs/final_corrected_v1_final20.json -- a new artifact; it overwrites nothing",
      "ALL EIGHT METRICS IDENTICAL (delta 0.0000 on P@1, P@3, P@5, MRR, Recall@5, Recall@10, Relevant_Top1, Answering@5). The fix removed 15 bogus USPSTF section anchors (15 -> 0) and 13 bogus ESVS ones (113 -> 100); NICE and every recommendation anchor were untouched. Chunks 1,764 -> 1,760 total, 1,004 -> 991 indexed, mean tokens 221.4 -> 223.7, still 0 over the limit, recommendation coverage 350 -> 351. Seven of twenty questions returned a differently identified top-1 chunk (chunk ids shift when boundaries move) with unchanged rank and relevance.",
      "None. The removed anchors only ever fragmented bibliography text, which the content classifier already excludes from the index, so retrieval never depended on them.",
      "PASSED THE GATE - V1 PROMOTED TO SHIPPED DEFAULT",
      "A defect can be real, worth fixing, and still immaterial to the result. Fixing it and measuring exactly zero change is a stronger statement than never having found it.",
      "Phase 11"),
]


def fmt_metrics(m: Any) -> str:
    if m == NA:
        return f"_{NA}_"
    if isinstance(m, dict) and all(k in m for k in KEYS):
        return " · ".join(f"{k} {m[k]}" for k in KEYS)
    return str(m)


def to_markdown(hist: dict) -> str:
    L: list[str] = []
    L.append("# Complete Experiment History\n")
    L.append("> Generated by `eval/build_experiment_history.py` from the preserved artifacts in "
             "`eval/runs/` and `eval/final_evaluation_results.json`. Metrics are read from those "
             "files, never retyped. Where a metric was never recorded the entry reads "
             f'"{NA}" — it is never estimated.\n')
    L.append("## Frozen gold standards\n")
    L.append("| set | path | SHA-256 |")
    L.append("|---|---|---|")
    for name, g in hist["gold_standards"].items():
        L.append(f"| `{name}` | `{g['path']}` | `{g['sha256']}` |")
    L.append(f"\n{hist['datasets_never_pooled']}\n")

    L.append("## Summary\n")
    L.append("| # | phase | experiment | decision |")
    L.append("|---:|---|---|---|")
    for e in hist["experiments"]:
        L.append(f"| {e['n']} | {e['phase']} | {e['name']} | **{e['decision']}** |")
    L.append("")

    for e in hist["experiments"]:
        L.append(f"\n---\n\n## {e['n']}. {e['name']}\n")
        L.append(f"**Phase:** {e['phase']} — **Decision: {e['decision']}**\n")
        L.append(f"**Problem.** {e['problem']}\n")
        L.append(f"**Hypothesis.** {e['hypothesis']}\n")
        L.append(f"**Intervention.** {e['intervention']}\n")
        L.append(f"**Dataset.** {e['dataset']}\n")
        L.append("**Metrics.**\n")
        for k, v in e["metrics"].items():
            L.append(f"- `{k}` — {fmt_metrics(v)}")
        L.append(f"\n**Evidence.** {e['evidence']}\n")
        L.append(f"**Result.** {e['result']}\n")
        L.append(f"**Failure mode.** {e['failure_mode']}\n")
        L.append(f"**Lesson learned.** {e['lesson_learned']}\n")

    if hist.get("final_recommendation"):
        L.append("\n---\n\n## Final recommendation\n")
        L.append(hist["final_recommendation"])
    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    hist = {
        "schema_version": "1.0",
        "generated_by": "eval/build_experiment_history.py",
        "note": ("Metrics are read directly from the preserved artifacts in eval/runs/ and "
                 "eval/final_evaluation_results.json. Where a metric was never recorded the value is "
                 "the literal string below; it is never estimated."),
        "unavailable_marker": NA,
        "metric_keys": list(KEYS),
        "gold_standards": {
            "original10": {"path": "eval/gold_standard.json",
                           "sha256": hashlib.sha256((EVAL_DIR / "gold_standard.json").read_bytes()).hexdigest()},
            "heldout18": {"path": "eval/gold_standard_heldout.json",
                          "sha256": hashlib.sha256((EVAL_DIR / "gold_standard_heldout.json").read_bytes()).hexdigest()},
            "final20": {"path": "eval/gold_standard_final20.json",
                        "sha256": hashlib.sha256((EVAL_DIR / "gold_standard_final20.json").read_bytes()).hexdigest()},
        },
        "datasets_never_pooled": ("original10, heldout18 and final20 use different questions, different "
                                  "answer passages and different difficulty. They are reported separately "
                                  "and must never be averaged together."),
        "experiments": EXPERIMENTS,
    }
    rec_path = EVAL_DIR / "final_recommendation.md"
    if rec_path.exists():
        hist["final_recommendation"] = rec_path.read_text(encoding="utf-8")

    out = EVAL_DIR / "experiment_history.json"
    out.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(EXPERIMENTS)} experiments)")

    md_path = ROOT / "docs" / "experiment_history.md"
    md_path.write_text(to_markdown(hist), encoding="utf-8")
    print(f"wrote {md_path}")

    missing = [(e["n"], e["name"], k) for e in EXPERIMENTS
               for k, v in e["metrics"].items() if v == NA]
    print(f"\nmetric slots marked unavailable: {len(missing)}")
    for n, name, k in missing:
        print(f"  #{n} {name[:48]:<50} [{k}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
