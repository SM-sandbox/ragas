# Reranker Impact Analysis

**Date:** December 15, 2025  
**Category:** Foundational  
**Status:** ✅ Complete

---

## 1. Objective

Quantify the impact of the Google Ranking API reranker on retrieval quality and determine if the latency cost is justified.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Initial Retrieval | Top 100 documents (hybrid 50/50 dense/sparse) |
| Reranker | Google Ranking API |
| Final Output | Top 12 documents for LLM generation |
| Embedding | gemini-embedding-001 (1536 dim, RETRIEVAL_QUERY) |
| Test Size | 224 questions × 3 runs |

---

## 3. Results

### MRR Improvement

| Stage | MRR@12 | Improvement |
|-------|--------|-------------|
| Pre-Rerank | 0.710 | Baseline |
| **Post-Rerank** | **0.782** | **+10%** |

### Document Reshuffling

| Metric | Value |
|--------|-------|
| Top 12 Overlap | 2.8/12 avg (**23%**) |
| Docs Promoted (13-100 → top 12) | 2.0 per query |
| Docs Demoted (top 12 → out) | 2.1 per query |

### Overlap Distribution

| Overlap | Queries | Percentage |
|---------|---------|------------|
| 0-3 docs overlap | 163 | **72.8%** |
| 4-6 docs overlap | 56 | 25.0% |
| 7+ docs overlap | 5 | 2.2% |

### Latency Cost

| Phase | Time |
|-------|------|
| Reranking | 0.196s avg |

---

## 4. Key Findings

1. **Significant quality improvement** - MRR improves by 10% (0.710 → 0.782).

2. **Aggressive reshuffling** - Only 23% of initial top-12 documents survive into the final top-12. The reranker is doing substantial work.

3. **Promotes from deep in the list** - Documents from positions 13-100 are regularly promoted into the final top-12, validating the strategy of retrieving 100 documents initially.

4. **73% of queries see major changes** - Most queries have 0-3 documents overlap between pre and post reranking.

5. **Low latency cost** - 0.19s per query is acceptable given the quality improvement.

---

## 5. Recommendation

✅ **Keep reranking enabled.** The 10% MRR improvement justifies the 0.19s latency cost.

Consider increasing initial retrieval from 100 to 150-200 documents, since the reranker actively promotes from the full retrieval pool.

---

## 6. Related Reports

- [Embedding Model Comparison](02_Embedding_Model_Comparison.md)
- [Context Size Sweep](../experiments/Context_Size_Sweep.md)
