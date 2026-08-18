# Generation evaluation

- eval set: `aaa-generation-eval-v1` (1 questions run)
- provider / model: **groq** / `openai/gpt-oss-120b`
- top_k 3, score floor 0.75, temperature 0.0
- run (UTC): 2026-08-18T11:45:21.540689+00:00

**0/1 passed the mechanical checks.** 0 refused, 1 passed the citation validator, 0 fabricated citation(s) in total.

| id | category | expected | outcome | chunks used | validator | checks | notes |
|---|---|---|---|---|---|---|---|
| G8 | conflicting | answer | answered | 3/3 | ok | FAIL | conflict_positions |

Per-category:

| category | passed / n |
|---|---|
| conflicting | 0 / 1 |

Mechanical checks only. Grounding and citation accuracy still need a human read; `manual_review` in the JSON carries the resolved citations and the chunks that were sent for each question.
