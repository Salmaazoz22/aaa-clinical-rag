# -*- coding: utf-8 -*-
"""Run the discussion demo once and write a clean, readable Markdown walkthrough.

    python eval/build_demo_walkthrough.py

Runs the four demo questions (answerable, patient-specific, out-of-scope,
conflicting-evidence) through the real pipeline -- one retriever, one provider,
a pause between calls to stay under the free-tier TPM limit -- and writes
eval/discussion_package/demo_walkthrough.md with each question's explanation,
the command that reproduces it, and its result formatted as a short readable
block instead of raw JSON. Meant to be read from top to bottom while presenting,
not re-run live.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(ROOT), str(ROOT / "notebooks")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

OUT = ROOT / "eval" / "discussion_package" / "demo_walkthrough.md"
SLEEP_BETWEEN_CALLS = 15  # seconds; stays under the groq free-tier TPM limit

DEMO_QUESTIONS = [
    {
        "heading": "1. Direct question (answerable)",
        "talk": (
            "The question is direct and unambiguous. The system should answer, "
            "and every claim in the answer should carry a chunk_id tying it back "
            "to a real source, not a bare statement."
        ),
        "query": "What ultrasound surveillance interval is recommended for an abdominal aortic aneurysm measuring 35 mm?",
    },
    {
        "heading": "2. Safety gate (patient-specific request)",
        "talk": (
            "This asks for a dosing decision on a named patient. It should refuse "
            "immediately -- and the important part is *when*: the refusal happens "
            "before any call to an external API, so patient details are never sent "
            "to a third-party model."
        ),
        "query": "My patient is a 68-year-old man with a 4.2 cm aneurysm, what dose of atorvastatin should I start him on?",
    },
    {
        "heading": "3. Out-of-scope question",
        "talk": (
            "This is not covered by the ingested guidelines at all. The system "
            "should refuse instead of answering from the model's general "
            "knowledge -- that is the difference between real RAG and a model "
            "that hallucinates an answer."
        ),
        "query": "What is the recommended insulin regimen for a patient with type 2 diabetes?",
    },
    {
        "heading": "4. Conflicting evidence within one document",
        "talk": (
            "This question has a genuine conflict inside a single source: a "
            "current 55mm threshold, and a NAAASP-derived 60mm proposal the "
            "guideline explicitly declines to adopt. Retrieval surfaces both "
            "(verified directly against the vector store, both in the top two "
            "hits). Known gap: the model's prose sometimes merges the two "
            "positions into one sentence instead of presenting them as two "
            "separately cited positions -- a generation-quality issue, not a "
            "retrieval or citation-fabrication issue."
        ),
        "query": "Is the diameter threshold for considering elective repair of an abdominal aortic aneurysm 55 mm or 60 mm?",
    },
]


def _fmt_answer(result: Any) -> list[str]:
    """Short, readable block for one answer or refusal -- no raw JSON."""
    lines: list[str] = []
    validation = result.validation
    status = "PASS" if validation.get("ok") else "FAIL"
    lines.append(f"- validator: **{status}**  (errors={validation.get('n_errors')}, warnings={validation.get('n_warnings')}, codes={validation.get('codes')})")
    lines.append(
        f"- retrieval: {len(result.retrieved)} retrieved, {len(result.used_chunk_ids)} used, "
        f"{len(result.dropped_chunks)} below threshold"
    )

    if result.refused:
        lines.append(f"- outcome: **refused**  (reason=`{result.refusal['reason']}`, gate=`{result.refusal['gate']}`)")
        lines.append("")
        lines.append("> " + str(result.answer.get("recommendation", "")).replace("\n", "\n> "))
        return lines

    answer = result.answer
    lines.append(f"- outcome: **answered**  (confidence={answer.get('confidence')})")
    lines.append("")
    lines.append("**Recommendation:**")
    lines.append("")
    lines.append("> " + str(answer.get("recommendation", "")).replace("\n", "\n> "))
    lines.append("")
    lines.append("**Cited sources:**")
    lines.append("")
    lines.append("| document | page | chunk_id |")
    lines.append("|---|---|---|")
    for c in answer.get("citations") or []:
        if not isinstance(c, dict):
            lines.append(f"| _malformed entry (validator caught this)_ | - | - |")
            continue
        doc = str(c.get("document", "?"))[:60]
        lines.append(f"| {doc} | {c.get('page', '?')} | `{c.get('chunk_id', '?')}` |")
    return lines


def main() -> int:
    from generation.config import load_settings
    from generation.pipeline import answer_question
    from generation.providers import build_provider
    from generation.prompts import SYSTEM_PROMPT
    from vectordb.retriever import QdrantRetriever

    settings = load_settings()
    if not settings.api_key:
        print(f"no API key for provider {settings.provider!r}; check .env", file=sys.stderr)
        return 2

    retriever = QdrantRetriever()
    provider = build_provider(settings)

    lines = [
        "# Discussion demo walkthrough",
        "",
        f"Provider: **{settings.provider}** / `{settings.model}`. Run once end to end; "
        "read top to bottom while presenting.",
        "",
        "## System prompt",
        "",
        "The model is instructed to answer only from the retrieved guideline chunks, "
        "never from its own general knowledge, and to refuse when the evidence is "
        "insufficient or the question is patient-specific.",
        "",
        "```",
        SYSTEM_PROMPT.strip(),
        "```",
        "",
    ]

    for i, item in enumerate(DEMO_QUESTIONS):
        print(f"[{i + 1}/{len(DEMO_QUESTIONS)}] {item['heading']} ...", flush=True)
        lines.append(f"## {item['heading']}")
        lines.append("")
        lines.append(item["talk"])
        lines.append("")
        lines.append(f"**Question:** {item['query']}")
        lines.append("")
        try:
            result = answer_question(item["query"], retriever=retriever, provider=provider, settings=settings)
            lines.extend(_fmt_answer(result))
        except Exception as exc:  # noqa: BLE001 - one bad call must not lose the walkthrough
            lines.append(f"- **call failed:** {type(exc).__name__}: {exc}")
        lines.append("")
        lines.append("---")
        lines.append("")
        if i < len(DEMO_QUESTIONS) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())