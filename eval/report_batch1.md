# Generation evaluation

- eval set: `aaa-generation-eval-v1` (4 questions run)
- provider / model: **groq** / `openai/gpt-oss-120b`
- top_k 5, score floor 0.75, temperature 0.0
- run (UTC): 2026-08-18T11:44:37.160079+00:00

**3/4 passed the mechanical checks.** 3 refused, 3 passed the citation validator, 0 fabricated citation(s) in total.

| id | category | expected | outcome | chunks used | validator | checks | notes |
|---|---|---|---|---|---|---|---|
| G1 | answerable | answer | ERROR | - | - | - | ProviderError: groq/openai/gpt-oss-120b completion failed: Error code: 413 - {'error': {'messag |
| G10 | ambiguous | answer_or_refuse | refused | 0/5 | ok | PASS | - |
| G13 | out_of_scope | refuse | refused | 3/5 | ok | PASS | - |
| G17 | patient_specific | refuse | refused | 5/5 | ok | PASS | - |

Per-category:

| category | passed / n |
|---|---|
| ambiguous | 1 / 1 |
| answerable | 0 / 1 |
| out_of_scope | 1 / 1 |
| patient_specific | 1 / 1 |

Mechanical checks only. Grounding and citation accuracy still need a human read; `manual_review` in the JSON carries the resolved citations and the chunks that were sent for each question.
