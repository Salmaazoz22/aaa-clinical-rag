# -*- coding: utf-8 -*-
"""Turn a generation-eval run into a fast side-by-side citation review packet.

    python eval/manual_citation_review.py
    python eval/manual_citation_review.py --results eval/generation_eval_results.json

The mechanical checks in `run_generation_eval.py` confirm a citation points to a
real chunk. They cannot confirm the claim's meaning is actually contained in that
chunk's text -- that needs a person to read both side by side. This script does
the fetching so the person doesn't have to: it pulls every cited chunk's full
text directly from Qdrant by chunk_id (`QdrantRetriever.get_by_chunk_ids`, exact
lookup, no search) and lays each claim next to its source in one Markdown file.

Refusals are skipped (no claims to check). Hallucinated citations already
reported by the validator are marked as such instead of being looked up, since
a fabricated chunk_id resolves to nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This script and its artifacts live together in eval/generation/, so the paths
# are derived from this file's own directory rather than from eval/ root.
GEN_DIR = Path(__file__).resolve().parent
RESULTS = GEN_DIR / "generation_eval_results.json"
OUT = GEN_DIR / "manual_citation_review.md"


def _claims_for(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Claim/chunk_id pairs for one answered question, empty for refusals."""
    manual_review = record.get("manual_review") or {}
    return [c for c in manual_review.get("claims") or [] if c.get("chunk_id")]


def build_report(results: dict[str, Any], retriever: Any) -> str:
    records = [r for r in results.get("records", []) if "error" not in r]
    answered = [r for r in records if not r.get("result", {}).get("refused")]

    all_chunk_ids = {
        claim["chunk_id"]
        for record in answered
        for claim in _claims_for(record)
    }
    chunks = retriever.get_by_chunk_ids(sorted(all_chunk_ids))

    lines = [
        "# Manual citation review",
        "",
        f"{len(answered)} answered question(s) out of {len(records)} run. "
        "For each claim below: read it next to its cited chunk and mark it.",
        "",
        "Mark each as `[x] MATCH` or `[x] MISMATCH` and add a one-line note if it mismatches.",
        "",
    ]

    for record in answered:
        question = record["question"]
        result = record["result"]
        hallucinated = set(result.get("validation", {}).get("hallucinated_chunk_ids") or [])
        claims = _claims_for(record)

        lines += [
            f"## {question['id']} -- {question['question']}",
            "",
            f"Category: {question['category']}  |  Recommendation: "
            f"{result.get('answer', {}).get('recommendation', '(none)')}",
            "",
        ]

        if not claims:
            lines += ["_No cited claims to review._", ""]
            continue

        for i, claim in enumerate(claims, start=1):
            chunk_id = claim["chunk_id"]
            lines.append(f"**Claim {i}:** {claim.get('claim')}")
            lines.append("")
            if chunk_id in hallucinated:
                lines.append(f"- `{chunk_id}` -- FLAGGED BY VALIDATOR AS HALLUCINATED, do not look up")
            else:
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    lines.append(f"- `{chunk_id}` -- not found in the collection (check ingestion)")
                else:
                    where = f"{chunk.get('document') or '?'}, p.{chunk.get('page') or '?'}"
                    if chunk.get("section"):
                        where += f", {chunk['section']}"
                    lines.append(f"- Source ({where}):")
                    lines.append("")
                    text = (chunk.get("chunk_text") or "").strip() or "(empty chunk_text)"
                    lines.append("  > " + text.replace("\n", "\n  > "))
            lines.append("")
            lines.append("- [ ] MATCH   - [ ] MISMATCH   note: __________")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"no results file at {results_path}; run eval/run_generation_eval.py first", file=sys.stderr)
        return 2
    results = json.loads(results_path.read_text(encoding="utf-8"))

    from vectordb.retriever import QdrantRetriever

    retriever = QdrantRetriever()
    report = build_report(results, retriever)

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
