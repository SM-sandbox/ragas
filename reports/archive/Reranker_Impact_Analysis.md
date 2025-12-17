# Reranker Impact Analysis

**Date:** 2025-12-15  
**Test:** E2E Orchestrator Test (3 runs × 224 questions)

---

## 1. Overview

This analysis examines the impact of the Google Ranking API reranker on retrieval quality. We compare the initial top 12 documents from hybrid vector search against the top 12 documents after reranking.

### Configuration

| Component | Setting |
|-----------|---------|
| Initial Retrieval | Top 100 documents (hybrid 50/50 dense/sparse) |
| Reranker | Google Ranking API |
| Final Output | Top 12 documents for LLM generation |
| Embedding | gemini-embedding-001 (1536 dim, RETRIEVAL_QUERY) |

---

## 2. Key Findings

### 2.1 Overlap Between Pre-Rerank and Post-Rerank Top 12

| Metric | Value |
|--------|-------|
| **Top 12 Overlap** | 2.8/12 avg (**23%**) |
| **Top 10 Overlap** | 2.4/10 avg (**24%**) |

**Only ~23% of the initial top 12 documents survive into the final top 12 after reranking.**

### 2.2 Document Movement

| Metric | Value |
|--------|-------|
| Docs Promoted (13-100 → top 12) | 2.0 per query avg |
| Docs Demoted (top 12 → out) | 2.1 per query avg |

The reranker actively promotes documents from positions 13-100 into the final top 12, while demoting initially high-ranked documents.

### 2.3 Position Changes (for docs staying in top 12)

| Metric | Value |
|--------|-------|
| Avg position change | -0.26 (moved up slightly) |
| Std dev | 3.66 |

Documents that remain in the top 12 experience moderate position shuffling.

---

## 3. MRR Improvement

| Stage | MRR@12 |
|-------|--------|
| Pre-Rerank | 0.710 |
| Post-Rerank | 0.782 |
| **Improvement** | **+0.071 (+10%)** |

The reranker improves Mean Reciprocal Rank by 10%, meaning the correct document is ranked higher after reranking.

---

## 4. Overlap Distribution

How many of the initial top 12 documents appear in the final top 12:

| Overlap | Queries | Percentage |
|---------|---------|------------|
| 0/12 | 1 | 0.4% |
| 1/12 | 55 | 24.6% |
| 2/12 | 55 | 24.6% |
| 3/12 | 52 | 23.2% |
| **0-3 overlap** | **163** | **72.8%** |
| 4/12 | 26 | 11.6% |
| 5/12 | 23 | 10.3% |
| 6/12 | 7 | 3.1% |
| **4-6 overlap** | **56** | **25.0%** |
| 7/12 | 3 | 1.3% |
| 8/12 | 2 | 0.9% |
| **7+ overlap** | **5** | **2.2%** |

**73% of queries see massive reshuffling (0-3 docs overlap).**

---

## 5. Conclusions

1. **The reranker is doing significant work** — it's not just minor reordering. On average, only ~3 of the initial top 12 documents survive into the final top 12.

2. **MRR improves by 10%** — the reranker successfully promotes more relevant documents to higher positions.

3. **The reranker pulls from the full retrieval pool** — documents from positions 13-100 are regularly promoted into the final top 12, validating the strategy of retrieving 100 documents initially.

4. **Justifies the reranking step** — the 0.19s latency cost for reranking delivers measurable quality improvement.

---

## 6. Recommendations

1. **Keep reranking enabled** — the 10% MRR improvement justifies the latency cost.

2. **Consider increasing initial retrieval** — since the reranker actively promotes from positions 13-100, retrieving more candidates (e.g., 150-200) might yield further improvements.

3. **Monitor reranker consistency** — the massive reshuffling (73% of queries with 0-3 overlap) suggests the reranker has strong opinions; worth validating these are correct opinions via human evaluation.
