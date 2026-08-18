# Generation evaluation

- eval set: `aaa-generation-eval-v1` (1 questions run)
- provider / model: **groq** / `openai/gpt-oss-120b`
- top_k 5, score floor 0.75, temperature 0.0
- run (UTC): 2026-08-18T11:48:32.906506+00:00

**1/1 passed the mechanical checks.** 0 refused, 1 passed the citation validator, 0 fabricated citation(s) in total.

| id | category | expected | outcome | chunks used | validator | checks | notes |
|---|---|---|---|---|---|---|---|
| G1 | answerable | answer | answered | 5/5 | ok | PASS | - |

Per-category:

| category | passed / n |
|---|---|
| answerable | 1 / 1 |

Mechanical checks only. Grounding and citation accuracy still need a human read; `manual_review` in the JSON carries the resolved citations and the chunks that were sent for each question.
