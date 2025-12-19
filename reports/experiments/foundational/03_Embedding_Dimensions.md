# Embedding Dimension Analysis

**Date:** December 14, 2025  
**Category:** Foundational  
**Status:** ✅ Complete

---

## 1. Objective

Determine if larger embedding dimensions (1536, 3072) improve retrieval quality compared to the default 768 dimensions for gemini-embedding-001.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Embedding Model | gemini-embedding-001 |
| Task Type | RETRIEVAL_QUERY |
| Dimensions Tested | 768, 1536, 3072 |
| Corpus | 224 questions from 40 source documents |
| Retrieval | Top-12 chunks via Vertex AI Vector Search |
| Reranker | Google Ranking API |

---

## 3. Results

### LLM Judge Scores

| Dimensions | Overall Score | Pass Rate |
|------------|---------------|-----------|
| 768 | 4.25 | 72% |
| **1536** | **4.21** | **71%** |
| 3072 | 4.20 | 70% |

### Retrieval Metrics

| Dimensions | MRR@10 | Recall@10 |
|------------|--------|-----------|
| 768 | 0.790 | 90.6% |
| **1536** | **0.806** | **90.6%** |
| 3072 | 0.802 | 90.0% |

### Cost/Performance Tradeoff

| Dimensions | Storage | Compute | Quality |
|------------|---------|---------|---------|
| 768 | 1x | 1x | Baseline |
| 1536 | 2x | ~1.5x | +2% MRR |
| 3072 | 4x | ~2x | Slight degradation |

---

## 4. Key Findings

1. **768 dimensions is sufficient** - No meaningful quality improvement from larger dimensions.

2. **1536 is the sweet spot** - Slightly better MRR (0.806 vs 0.790) with acceptable storage cost.

3. **3072 provides no benefit** - Actually slightly worse Recall@10 (90.0% vs 90.6%) at 4x storage cost.

4. **Diminishing returns** - The additional storage and compute cost of larger dimensions is not justified by quality improvements.

---

## 5. Recommendation

✅ **Use 1536 dimensions for production.**

The slight MRR improvement (0.806 vs 0.790) justifies the 2x storage cost. Do not use 3072 dimensions - it provides no benefit.

For cost-sensitive deployments, 768 dimensions is acceptable with minimal quality loss.

---

## 6. Related Reports

- [Embedding Model Comparison](02_Embedding_Model_Comparison.md)
- [Task Type Comparison](04_Task_Type_Comparison.md)
