# -*- coding: utf-8 -*-
"""Run the generation-evaluation set and score what can be scored mechanically.

    python eval/run_generation_eval.py                       # configured provider
    python eval/run_generation_eval.py --provider groq        # fast iteration
    python eval/run_generation_eval.py --provider openrouter  # final answers
    python eval/run_generation_eval.py --ids G8,G14,G17       # a subset

Writes a NEW artifact pair and overwrites no existing evaluation file:

    eval/generation_eval_results.json    full audit record per question
    eval/generation_eval_report.md       human-readable summary table

This is a *generation* evaluation. It does not touch, recompute or compare
against the frozen retrieval gold standards, and its numbers are not comparable
to the retrieval metrics in eval/final_evaluation_results.json.

Two thirds of the specification's criteria are mechanical -- did it refuse when it
should, are the citations real, are the required facts stated -- and this script
scores those. The remaining criterion ("is each citation the chunk that actually
contains the claim") needs a person to read the chunk text, so the script emits
the evidence needed for that read rather than pretending to automate it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.config import load_settings  # noqa: E402
from generation.pipeline import answer_question  # noqa: E402
from generation.providers import ProviderError, build_provider  # noqa: E402
from generation.validator import answer_prose, conflict_position_count  # noqa: E402

# This script and its artifacts live together in eval/generation/, so the paths
# are derived from this file's own directory rather than from eval/ root.
GEN_DIR = Path(__file__).resolve().parent
EVAL_SET = GEN_DIR / "generation_eval_set.json"
RESULTS = GEN_DIR / "generation_eval_results.json"
REPORT = GEN_DIR / "generation_eval_report.md"


def full_text(answer: dict[str, Any]) -> str:
    """Everything the answer contains, quoted excerpts included."""
    return json.dumps(answer, ensure_ascii=False)


def score_question(question: dict[str, Any], result: Any) -> dict[str, Any]:
    """Apply the mechanical checks to one answered question."""
    answer = result.answer
    prose = answer_prose(answer).lower()
    everything = full_text(answer).lower()
    validation = result.validation
    checks: dict[str, Any] = {}

    expected = question.get("expected")
    refused = result.refused
    if expected == "answer":
        checks["refusal_expectation"] = {
            "pass": not refused,
            "detail": "expected an answer" + (", got a refusal" if refused else ", got an answer"),
        }
    elif expected == "refuse":
        checks["refusal_expectation"] = {
            "pass": refused,
            "detail": "expected a refusal" + (", got a refusal" if refused else ", got an answer"),
        }
    else:  # answer_or_refuse
        checks["refusal_expectation"] = {
            "pass": True,
            "detail": f"either outcome acceptable; got {'refusal' if refused else 'answer'}",
        }

    acceptable = question.get("acceptable_refusal_reasons")
    if refused and acceptable:
        reason = (result.refusal or {}).get("reason")
        checks["refusal_reason"] = {
            "pass": reason in acceptable,
            "detail": f"reason={reason!r}, acceptable={acceptable}",
        }

    checks["validator"] = {
        "pass": bool(validation.get("ok")),
        "detail": (
            f"{validation.get('n_errors')} error(s), {validation.get('n_warnings')} warning(s): "
            f"{validation.get('codes')}"
        ),
    }

    missing = [s for s in question.get("must_mention") or [] if s.lower() not in prose]
    if question.get("must_mention"):
        # Only meaningful when an answer was expected; a correct refusal cannot be
        # penalised for failing to state facts it correctly declined to state.
        applicable = not (refused and expected != "answer")
        checks["must_mention"] = {
            "pass": (not missing) if applicable else True,
            "detail": (
                f"missing from the answer's own prose: {missing}" if missing else "all present"
            )
            + ("" if applicable else " (not applicable: correctly refused)"),
            "missing": missing,
            "also_absent_from_excerpts": [s for s in missing if s.lower() not in everything],
        }

    forbidden = [s for s in question.get("must_not_mention") or [] if s.lower() in everything]
    if question.get("must_not_mention"):
        checks["must_not_mention"] = {
            "pass": not forbidden,
            "detail": f"present but forbidden: {forbidden}" if forbidden else "none present",
            "found": forbidden,
        }

    min_positions = question.get("min_conflict_positions") or 0
    if min_positions:
        found = conflict_position_count(answer)
        docs = result.documents_cited
        checks["conflict_positions"] = {
            "pass": found >= min_positions,
            "detail": f"{found} position(s) reported, need >= {min_positions}; documents cited: {docs}",
            "positions": found,
        }

    applicable_checks = [c for c in checks.values() if isinstance(c, dict) and "pass" in c]
    return {
        "checks": checks,
        "passed": all(c["pass"] for c in applicable_checks),
        "n_checks": len(applicable_checks),
        "n_failed": sum(1 for c in applicable_checks if not c["pass"]),
    }


def manual_review_packet(result: Any) -> dict[str, Any]:
    """What a person needs in order to judge grounding and citation accuracy."""
    return {
        "recommendation": result.answer.get("recommendation"),
        "confidence": result.answer.get("confidence"),
        "claims": [
            {"claim": b.get("claim"), "chunk_id": b.get("chunk_id")}
            for b in result.answer.get("supporting_evidence") or []
            if isinstance(b, dict)
        ],
        "citations_resolved": result.citations_resolved,
        "documents_cited": result.documents_cited,
        "chunks_sent": result.used_chunk_ids,
        "chunks_dropped_below_threshold": result.dropped_chunks,
        "questions_for_the_reviewer": [
            "Does every sentence of the recommendation trace to one of the cited chunks?",
            "Is each citation the chunk that actually contains its claim?",
            "If this is a refusal: does it name what was found, what is missing, and what would answer it?",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the generation-evaluation set.")
    ap.add_argument("--provider", choices=("groq", "openrouter"), default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated question ids to run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between calls (free tiers rate-limit)")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()

    spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.ids:
        wanted = {q.strip() for q in args.ids.split(",") if q.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("no questions selected", file=sys.stderr)
        return 2

    settings = load_settings(provider=args.provider)
    if not settings.api_key:
        print(
            f"No API key for provider {settings.provider!r}. Set the matching variable in .env "
            f"(see .env.example).",
            file=sys.stderr,
        )
        return 2

    # One retriever and one provider for the whole run: the embedding model loads
    # once, and the run is one configuration rather than a mixture.
    from vectordb.retriever import QdrantRetriever

    retriever = QdrantRetriever()
    provider = build_provider(settings)

    print(f"provider={settings.provider} model={settings.model} "
          f"top_k={args.top_k or settings.top_k} threshold={args.threshold or settings.score_threshold}")

    records: list[dict[str, Any]] = []
    for i, question in enumerate(questions, start=1):
        qid, text = question["id"], question["question"]
        print(f"[{i}/{len(questions)}] {qid} ({question['category']}) ...", end=" ", flush=True)
        record: dict[str, Any] = {"question": question}
        try:
            result = answer_question(
                text,
                retriever=retriever,
                provider=provider,
                settings=settings,
                top_k=args.top_k,
                threshold=args.threshold,
            )
        except ProviderError as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["scoring"] = {"passed": False, "n_checks": 0, "n_failed": 1, "checks": {}}
            records.append(record)
            print("PROVIDER ERROR")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad question must not lose the run
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["scoring"] = {"passed": False, "n_checks": 0, "n_failed": 1, "checks": {}}
            records.append(record)
            print(f"ERROR: {type(exc).__name__}")
            continue

        scoring = score_question(question, result)
        record["result"] = result.to_dict()
        record["scoring"] = scoring
        record["manual_review"] = manual_review_packet(result)
        records.append(record)
        print(
            ("PASS" if scoring["passed"] else "FAIL")
            + f"  ({'refused' if result.refused else 'answered'}, "
            + f"{len(result.used_chunk_ids)}/{len(result.retrieved)} chunks used, "
            + f"validator {'ok' if result.validation.get('ok') else 'FAIL'})"
        )

        if args.sleep and i < len(questions):
            time.sleep(args.sleep)

    by_category: dict[str, dict[str, int]] = {}
    for record in records:
        category = record["question"]["category"]
        bucket = by_category.setdefault(category, {"n": 0, "passed": 0})
        bucket["n"] += 1
        bucket["passed"] += 1 if record["scoring"]["passed"] else 0

    summary = {
        "eval_set_id": spec["id"],
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "provider": settings.provider,
        "model": settings.model,
        "top_k": args.top_k or settings.top_k,
        "score_threshold": args.threshold or settings.score_threshold,
        "temperature": settings.temperature,
        "n_questions": len(records),
        "n_passed": sum(1 for r in records if r["scoring"]["passed"]),
        "n_provider_errors": sum(1 for r in records if "error" in r),
        "n_refused": sum(1 for r in records if r.get("result", {}).get("refused")),
        "n_validator_ok": sum(
            1 for r in records if r.get("result", {}).get("validation", {}).get("ok")
        ),
        "n_hallucinated_citations": sum(
            len(r.get("result", {}).get("validation", {}).get("hallucinated_chunk_ids") or [])
            for r in records
        ),
        "by_category": by_category,
    }

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Generation evaluation",
        "",
        f"- eval set: `{spec['id']}` ({summary['n_questions']} questions run)",
        f"- provider / model: **{summary['provider']}** / `{summary['model']}`",
        f"- top_k {summary['top_k']}, score floor {summary['score_threshold']}, temperature {summary['temperature']}",
        f"- run (UTC): {summary['run_utc']}",
        "",
        f"**{summary['n_passed']}/{summary['n_questions']} passed the mechanical checks.** "
        f"{summary['n_refused']} refused, {summary['n_validator_ok']} passed the citation validator, "
        f"{summary['n_hallucinated_citations']} fabricated citation(s) in total.",
        "",
        "| id | category | expected | outcome | chunks used | validator | checks | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        question = record["question"]
        scoring = record["scoring"]
        if "error" in record:
            lines.append(
                f"| {question['id']} | {question['category']} | {question['expected']} | "
                f"ERROR | - | - | - | {record['error']['type']}: {record['error']['message'][:80]} |"
            )
            continue
        result = record["result"]
        failed = [name for name, check in scoring["checks"].items() if not check.get("pass", True)]
        lines.append(
            f"| {question['id']} | {question['category']} | {question['expected']} | "
            f"{'refused' if result['refused'] else 'answered'} | "
            f"{result['retrieval']['n_used']}/{result['retrieval']['n_retrieved']} | "
            f"{'ok' if result['validation']['ok'] else 'FAIL'} | "
            f"{'PASS' if scoring['passed'] else 'FAIL'} | "
            f"{', '.join(failed) if failed else '-'} |"
        )
    lines += [
        "",
        "Per-category:",
        "",
        "| category | passed / n |",
        "|---|---|",
    ]
    for category, bucket in sorted(by_category.items()):
        lines.append(f"| {category} | {bucket['passed']} / {bucket['n']} |")
    lines += [
        "",
        "Mechanical checks only. Grounding and citation accuracy still need a human read; "
        "`manual_review` in the JSON carries the resolved citations and the chunks that were sent "
        "for each question.",
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"{summary['n_passed']}/{summary['n_questions']} passed  "
          f"({summary['n_refused']} refused, {summary['n_hallucinated_citations']} fabricated citations)")
    print(f"wrote {out_path}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
