# Generation evaluation -- final 6-question demo run

**5/6 passed the mechanical checks.** 4 refused, 0 fabricated citation(s).

| id | category | expected | outcome | validator | checks | notes |
|---|---|---|---|---|---|---|
| G1 | answerable | answer | answered | ok | PASS | - |
| G8 | conflicting | answer | answered | ok | FAIL | conflict_positions |
| G10 | ambiguous | answer_or_refuse | refused | ok | PASS | - |
| G13 | out_of_scope | refuse | refused | ok | PASS | - |
| G14 | out_of_scope | refuse | refused | ok | PASS | - |
| G17 | patient_specific | refuse | refused | ok | PASS | - |

Per-category:

| category | passed / n |
|---|---|
| ambiguous | 1 / 1 |
| answerable | 1 / 1 |
| conflicting | 0 / 1 |
| out_of_scope | 2 / 2 |
| patient_specific | 1 / 1 |

Known gap: G8 (conflicting-evidence case) -- retrieval correctly surfaces both the current 55mm threshold and the NAAASP-derived 60mm proposal ESVS declined to adopt (confirmed directly against Qdrant, both in the top 2 hits). The model's prose merges them into one statement instead of presenting two distinct cited positions, so the mechanical conflict-position check fails. This is a generation-quality gap, not a retrieval or citation-fabrication issue.
