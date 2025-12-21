# Corpus Creation Log

This log tracks the creation and evolution of QA corpora for the BFAI Eval Suite.

---

## Naming Convention

**Format:** `QA_{client}_{tier}_v{version}__q{count}.json`

| Component | Values | Description |
|-----------|--------|-------------|
| **client** | `BFAI`, `ClientABC`, ... | Client/project identifier |
| **tier** | `silver`, `gold` | Silver = raw generated, Gold = quality-filtered |
| **version** | `v1-0`, `v1-1`, `v2-0` | Version (hyphen for GCS compatibility) |
| **count** | `q458`, `q500`, ... | Question count with 'q' prefix |

**Example:** `QA_BFAI_gold_v1-0__q458.json`

**Question ID Format:** `{hop}_{difficulty}_{number}`

| Prefix | Meaning |
|--------|---------|
| `sh_easy_001` | Single-hop, easy, question 1 |
| `sh_med_015` | Single-hop, medium, question 15 |
| `sh_hard_042` | Single-hop, hard, question 42 |
| `mh_easy_001` | Multi-hop, easy, question 1 |
| `mh_med_015` | Multi-hop, medium, question 15 |
| `mh_hard_042` | Multi-hop, hard, question 42 |

---

## Corpus History

### qa_corpus_gold_v1_458.json

**Created:** 2025-12-16  
**Questions:** 458  
**Status:** Active - Current gold standard

**Pipeline:**
1. Generated ~500 raw questions from 66 technical documents (Silver)
2. Scored each question for relevance (1-5 scale)
3. Filtered to relevance >= 4 (Critical + Relevant)
4. Result: 458 gold standard questions

**Distribution:**

| Type | Easy | Medium | Hard | Total |
|------|------|--------|------|-------|
| Single-hop | 88 | 78 | 56 | 222 |
| Multi-hop | 73 | 83 | 80 | 236 |
| **Total** | 161 | 161 | 136 | **458** |

**Evaluation Results:**
- Pass rate: 96.1% (P@25 config)
- Failures: 18 (3.9%)
- See: `reports/gold_standard_eval/Gold_Standard_Comparison_Report.md`

**Question ID Update:** 2025-12-17
- Changed from generic IDs (`q_0003`) to descriptive IDs (`sh_easy_001`)
- Mapping saved: `corpus/question_id_mapping.json`

---

## Archived Corpora

| File | Questions | Notes |
|------|-----------|-------|
| `qa_corpus_v2.json` | 453 | Intermediate version |
| `qa_corpus_gold_100.json` | 100 | Subset for testing |
| `qa_corpus_200.json` | 200 | Early generation |
| `qa_corpus_from_kg.json` | - | Knowledge graph based |

---

## Usage

**For new evaluations:**
```bash
# Use the current gold corpus
python scripts/run_gold_eval.py --corpus corpus/qa_corpus_gold_v1_458.json
```

**For client deployments:**
1. Copy this pipeline structure
2. Generate silver corpus from client documents
3. Score and filter to gold
4. Apply question ID nomenclature

---

*Last updated: 2025-12-17*
