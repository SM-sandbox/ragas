# Embedding Dimension Test Report

**Date:** 2024-12-14  
**Experiment:** `experiments/2024-12-14_embedding_dimension_test/`

---

## 1. Objective

Test whether larger embedding dimensions (1536, 3072) improve retrieval quality compared to the default 768 dimensions for gemini-embedding-001.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Embedding Model | gemini-embedding-001 |
| Task Type | RETRIEVAL_QUERY |
| Dimensions Tested | 768, 1536, 3072 |
| Corpus | 224 questions from 40 source documents |
| Retrieval | Top-12 chunks via Vertex AI Vector Search |
| Generation | Gemini 2.5 Flash |
| Judge | Gemini 2.5 Flash |
| Reranker | Google Ranking API |

---

## 3. Results

### 3.1 LLM Judge Scores

| Dimensions | Overall Score | Pass Rate |
|------------|---------------|-----------|
| 768 | 4.25 | 72% |
| 1536 | 4.21 | 71% |
| 3072 | 4.20 | 70% |

### 3.2 Retrieval Metrics (from Semantic Retrieval Test)

| Config | MRR@10 | Recall@10 |
|--------|--------|-----------|
| gemini-768-RETRIEVAL_QUERY | 0.790 | 90.6% |
| gemini-1536-RETRIEVAL_QUERY | 0.806 | 90.6% |
| gemini-3072-RETRIEVAL_QUERY | 0.802 | 90.0% |

---

## 4. Key Findings

### 4.1 Larger Dimensions Do NOT Improve Performance

**768 dimensions performs equally well or slightly better than 1536 or 3072.**

| Finding | Details |
|---------|---------|
| LLM Judge | 768 dim scores 4.25 vs 1536 dim scores 4.21 (-0.04) |
| Pass Rate | 768 dim: 72% vs 1536 dim: 71% (-1%) |
| Recall@10 | All dimensions achieve ~90.6% (no difference) |
| MRR@10 | 1536 slightly better (0.806 vs 0.790) but negligible |

### 4.2 Diminishing Returns

- Moving from 768 → 1536: No meaningful improvement
- Moving from 1536 → 3072: Slight degradation (90.0% vs 90.6% Recall@10)

### 4.3 Cost/Performance Tradeoff

| Dimensions | Storage | Compute | Quality |
|------------|---------|---------|---------|
| 768 | 1x | 1x | Baseline |
| 1536 | 2x | ~1.5x | No improvement |
| 3072 | 4x | ~2x | Slight degradation |

---

## 5. Conclusions

1. **768 dimensions is sufficient** for gemini-embedding-001 with RETRIEVAL_QUERY task type

2. **No benefit from larger dimensions** — the additional storage and compute cost is not justified

3. **Recommendation:** Use 768 dimensions for production deployments

---

## 6. Note on Later Findings

In subsequent testing (2025-12-15 E2E Orchestrator Test), we used 1536 dimensions and achieved:
- Recall@10: 93.3%
- MRR@10: 0.843

The improvement was due to **hybrid search (50/50 dense/sparse)** and **reranking**, not the dimension increase. The dimension test here used pure vector search without hybrid.

---

## 7. Data Files

- `experiments/2024-12-14_embedding_dimension_test/data/`
- `experiments/2024-12-14_embedding_dimension_test/README.md`
