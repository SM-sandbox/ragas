# Embedding Model Comparison

**Date:** December 15, 2025  
**Category:** Foundational  
**Status:** ✅ Complete

---

## 1. Objective

Determine which embedding model produces the best retrieval quality for the BrightFox technical document corpus.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Test Corpus | 160 questions with known source documents |
| Retrieval Method | Pure semantic search (no hybrid) |
| Top-K | 100 |
| Reranker | Disabled (testing embeddings only) |

### Models Tested

| Model | Dimensions | Task Type | Platform |
|-------|------------|-----------|----------|
| text-embedding-005 | 768 | None | GCP |
| gemini-embedding-001 | 768 | RETRIEVAL_QUERY | GCP |
| gemini-embedding-001 | 1536 | RETRIEVAL_QUERY | GCP |
| gemini-embedding-001 | 3072 | RETRIEVAL_QUERY | GCP |
| text-embedding-3-large | 3072 | None | Azure |

---

## 3. Results

### Recall@K (Higher is Better)

| Model | R@5 | R@10 | R@20 | R@50 | R@100 |
|-------|-----|------|------|------|-------|
| **gemini-1536-RETRIEVAL_QUERY** | **86.9%** | **90.6%** | **93.8%** | 96.2% | 97.5% |
| gemini-768-RETRIEVAL_QUERY | 84.4% | 90.6% | 92.5% | 96.9% | 97.5% |
| text-embedding-005 | 81.9% | 86.9% | 91.9% | 96.9% | 99.4% |
| gemini-3072-RETRIEVAL_QUERY | 86.2% | 90.0% | 92.5% | 95.6% | 97.5% |
| azure-text-embedding-3-large | 67.5% | 70.0% | 72.5% | 80.0% | 83.8% |

### MRR (Mean Reciprocal Rank)

| Model | MRR |
|-------|-----|
| **gemini-1536-RETRIEVAL_QUERY** | **0.806** |
| gemini-3072-RETRIEVAL_QUERY | 0.802 |
| gemini-768-RETRIEVAL_QUERY | 0.801 |
| text-embedding-005 | 0.760 |
| azure-text-embedding-3-large | 0.608 |

### Latency

| Model | Avg Retrieval Time |
|-------|-------------------|
| text-embedding-005 | 0.172s |
| gemini-768-RETRIEVAL_QUERY | 0.181s |
| gemini-1536-RETRIEVAL_QUERY | 0.192s |
| gemini-3072-RETRIEVAL_QUERY | 0.192s |

---

## 4. Key Findings

1. **gemini-embedding-001 with RETRIEVAL_QUERY wins** - Best MRR (0.806) and tied for best Recall@10 (90.6%).

2. **Task type matters significantly** - RETRIEVAL_QUERY improves Recall@10 by 7.5% over SEMANTIC_SIMILARITY (see Task Type report).

3. **1536 dimensions is the sweet spot** - Slightly better MRR than 768, no benefit from 3072.

4. **Azure underperforms** - 70% Recall@10 vs 90.6% for GCP. Likely due to different document set and no hybrid search.

5. **Latency is consistent** - All models perform in ~0.17-0.19s range.

---

## 5. Recommendation

✅ **Use gemini-embedding-001 with 1536 dimensions and RETRIEVAL_QUERY task type.**

This configuration provides:
- Best retrieval quality (MRR 0.806, Recall@10 90.6%)
- Reasonable latency (~0.19s)
- Good balance of storage cost vs quality

---

## 6. Related Reports

- [Embedding Dimensions](03_Embedding_Dimensions.md)
- [Task Type Comparison](04_Task_Type_Comparison.md)
- [Orchestrator vs Direct](01_Orchestrator_vs_Direct.md)
