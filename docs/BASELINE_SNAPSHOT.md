# Verified Baseline Snapshot — Project A (AAA Clinical RAG)

Recorded before any experimental change. Every value below was **read from the repository**,
not copied from the README. Verification was read-only: no repository file was modified to
produce this record.

## 1. Test suite

```
python -m pytest tests -q
28 passed in 95.79s
```

**28/28 passing.** No test was modified, skipped, or weakened.

## 2. Corpus

| Item | Verified value |
|---|---:|
| Source PDFs | 4 |
| Total pages (page records in `pages.json`) | 249 |
| Extracted recommendations | 233 |

Pages per document: ESVS_2024 140 · NICE_NG156 53 · SVS_2018 48 · USPSTF_2019 8 → **249**.

## 3. Chunks

| Item | Verified value |
|---|---:|
| Total chunks (`data/chunks/chunks.json`) | 2,116 |
| Indexed chunks (`data/embeddings/embedded_chunks.json`) | 1,330 |
| Chunk-quality report status | PASS |
| Chunk IDs unique | yes |

Content-type breakdown of all 2,116 chunks:

| content_type | count |
|---|---:|
| clinical (indexed) | 1,330 |
| reference | 661 |
| toc | 103 |
| boilerplate | 14 |
| title_only | 8 |

Chunks per document: ESVS_2024 1,806 · NICE_NG156 158 · USPSTF_2019 104 · SVS_2018 48.
Indexed per document: ESVS_2024 1,095 · NICE_NG156 118 · USPSTF_2019 73 · SVS_2018 44.

## 4. Embeddings and index

| Item | Verified value |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Model token window | 256 |
| Vectors on disk | `(1330, 384)` float32 |
| Index / chunk alignment | 1,330 metadata records = 1,330 vectors ✅ |
| All indexed chunk_ids present in `chunks.json` | ✅ |
| Vector L2 norms | 0.99999988 – 1.00000012 (normalised) |
| Index type / metric | `numpy_cosine` / cosine |

## 5. Token safety

| Item | Verified value |
|---|---:|
| Max token count, all 2,116 chunks (stored) | 254 |
| Max token count, 1,330 indexed chunks (stored) | 254 |
| **Max token count re-tokenised independently** | **254** |
| Chunks over the 256-token limit | **0** |
| Stored `token_count` vs recomputed mismatches | **0** |

Token counts were recomputed from `chunk_text` with the model's own tokenizer rather than
trusting the stored field. They agree exactly on all 1,330 indexed chunks — no silent
truncation, no stale metadata.

## 6. Section-title coverage (baseline defect measurement)

| Item | Count | % of indexed |
|---|---:|---:|
| Indexed chunks with a `section_title` | 1,157 | 87.0% |
| Indexed chunks with `section_title = None` | 173 | 13.0% |

This confirms the previously audited "~13% of indexed chunks have no section title".
(The README states 1,150/1,330; the actual current build is **1,157/1,330**. The README is
stale by 7 chunks on this one line; every other README figure verified exactly.)

## 7. Baseline checksums (SHA-256)

Full-file digests of every relevant artifact. Any experiment can be verified as reverted by
re-computing these.

| File | SHA-256 |
|---|---|
| `notebooks/clinical_chunking.py` | `a37d072c88ca3ea0d145951ac64a957cf83a00b077cabc700e28ccaddc52c6b7` |
| `notebooks/clinical_preprocess.py` | `1e6c69aef5b91e3dfcb2604cbaa32782145cf1d1a1431a72d5a1dce11f1fd07d` |
| `notebooks/clinical_rag.py` | `436fd6a96007bcb70cc3ea9583278c94788593aa708b1af18f48c596fcea1a01` |
| `notebooks/01_aaa_pdf_parsing_and_cleaning.ipynb` | `7e2ba38d0ea3ad551926cf299d1d5d6c21e422e43c304fa81b878d1516c88860` |
| `notebooks/02_aaa_chunking_and_embeddings.ipynb` | `7ee86677defc14e364a23968ca2eabd1763f33201c5d45ee77e8c5e9925047ad` |
| `notebooks/03_aaa_retrieval.ipynb` | `a619744313911d5fa39636d425c2803c3456d0c82ca15992801d3f37a77790eb` |
| `tests/test_chunking.py` | `aed4619076432178f13ef284860705b50672f6acabc0f301c2888b4eb4b9ec6b` |
| `data/chunks/chunks.json` | `a07bc49a6b8b7fd81fe46ddb934f1e24c61a57c26997e7da8c2c4417cd357382` |
| `data/embeddings/embedded_chunks.json` | `4a27dacaeb43a86b5936c78a33307643ed5289273c919c307bb93c014c52cc76` |
| `data/embeddings/embeddings.npy` | `5893ee16c599ef65a9fc3ec458502352e8ee5f049ca27ccf2b544f3fa11a3272` |
| `data/embeddings/index_meta.json` | `14390925043fb3cfaae63e1615bc83b5d875058b0c6a36fb4e8dde0c65924a95` |
| `data/processed/document_metadata.json` | `e6c65990a93604f6408df230c2eb7216fd3f0f5a310cf80fe6192b9b602383b4` |
| `data/processed/extraction_report.json` | `2335fa4440e9925ffa3c32a6b8987b7f4249036eb995e33b51f00bf18362b3ea` |
| `data/processed/pages.json` | `e402e4070056f5c354798b6e0bdd9202737f7d588a2322b12f5bb346b2d52a06` |
| `data/processed/pages_df.parquet` | `167b28a3a4d2fb83e8e8b3face7f93bac12209f351c129461826a45a77774c39` |
| `data/processed/recommendations.json` | `f90122607bf393722b33e077d8999ea308543c737ac78170354718ba37aed941` |
| `data/pdfs/abdom-aortic-aneurysm-screening-final-rs.pdf` | `a813b94cf2f8f1434886683b6597cce54a9cd2cc0bf2ac7973a68a37f5923ac3` |
| `data/pdfs/abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf` | `3f6abbd1ca76d3e78bc80c8227f6b9cffc8712bb13cf58f81f43116a54ddff30` |
| `data/pdfs/ESVS_2024_AAA_Guidelines.pdf` | `ce2e0ad3ba0d8c6dd17ab9fd56b616ed813e1d6a3031c2eef5bb5099230dfd5b` |
| `data/pdfs/SVS_Guideline_AAA_Slides_0.pdf` | `d26d9de82f12541e7ca65a0e557d529a8cf69b43a7eb510f1b544c0cdfa3e212` |
| `README.md` | `e67d1c67ad0ecc35e4f22b544596c3dc63c3e72088bd9732b92e3e1a2d44b8ae` |
| `requirements.txt` | `c5072759e19b5e54340301d0a44a9f4884026c023f65391da0431bb86b53089b` |

## 8. Discrepancies found against the stated expectations

| Expected (as given) | Verified | Status |
|---|---|---|
| 4 PDFs | 4 | ✅ |
| 249 pages | 249 | ✅ |
| 2,116 chunks | 2,116 | ✅ |
| 1,330 indexed chunks | 1,330 | ✅ |
| all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | ✅ |
| 384 dimensions | 384 | ✅ |
| max chunk ≤ 254 tokens | 254 | ✅ |
| 0 chunks over 256 tokens | 0 | ✅ |
| 28/28 tests passing | 28/28 | ✅ |

Only stale README line: section-title coverage (1,150 stated vs 1,157 actual).

## 9. Evaluation harness status

**MISSING.** See the report accompanying this snapshot. No frozen gold standard, relevance
labels, or metric implementation (P@1/P@3/P@5/MRR/Recall@5/Recall@10) exists anywhere in this
repository. Experiments 1–4 are therefore **blocked** and have not been started: no baseline
retrieval metrics exist to compare against, and inventing a gold standard would violate the
comparability requirement.

The repository is at the **original, unmodified baseline**.
