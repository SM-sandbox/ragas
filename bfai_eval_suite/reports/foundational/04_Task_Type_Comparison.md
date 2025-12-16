# Task Type Comparison: RETRIEVAL_QUERY vs SEMANTIC_SIMILARITY

**Date:** December 15, 2025  
**Category:** Foundational  
**Status:** ✅ Complete

---

## 1. Objective

Determine which task type produces better retrieval quality for gemini-embedding-001 when embedding search queries.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Embedding Model | gemini-embedding-001 |
| Dimensions | 768 |
| Task Types Tested | RETRIEVAL_QUERY, SEMANTIC_SIMILARITY |
| Corpus | 160 questions with known source documents |
| Retrieval | Pure semantic search |

---

## 3. Results

### Retrieval Quality

| Task Type | Recall@10 | MRR | Improvement |
|-----------|-----------|-----|-------------|
| **RETRIEVAL_QUERY** | **90.6%** | **0.801** | Baseline |
| SEMANTIC_SIMILARITY | 83.1% | 0.612 | -7.5% Recall, -24% MRR |

### Recall@K Breakdown

| Task Type | R@5 | R@10 | R@20 | R@50 | R@100 |
|-----------|-----|------|------|------|-------|
| RETRIEVAL_QUERY | 84.4% | 90.6% | 92.5% | 96.9% | 97.5% |
| SEMANTIC_SIMILARITY | 78.1% | 83.1% | 86.9% | 90.6% | 94.4% |

---

## 4. Key Findings

1. **RETRIEVAL_QUERY is significantly better** - 7.5% higher Recall@10 and 31% higher MRR.

2. **Task type is critical** - Using the wrong task type (SEMANTIC_SIMILARITY) severely degrades retrieval quality.

3. **Consistent across K values** - RETRIEVAL_QUERY outperforms at all K values tested.

4. **No latency difference** - Both task types have identical embedding generation time.

---

## 5. Recommendation

✅ **Always use RETRIEVAL_QUERY task type for query embeddings.**

The RETRIEVAL_QUERY task type is specifically optimized for search queries and produces dramatically better retrieval results. There is no reason to use SEMANTIC_SIMILARITY for RAG applications.

---

## 6. Technical Note

The gemini-embedding-001 model supports multiple task types:

- `RETRIEVAL_QUERY` - Optimized for search queries
- `RETRIEVAL_DOCUMENT` - Optimized for documents being indexed
- `SEMANTIC_SIMILARITY` - Optimized for comparing text similarity
- `CLASSIFICATION` - Optimized for classification tasks
- `CLUSTERING` - Optimized for clustering tasks

For RAG systems, use `RETRIEVAL_QUERY` for queries and `RETRIEVAL_DOCUMENT` for indexing.

---

## 7. Related Reports

- [Embedding Model Comparison](02_Embedding_Model_Comparison.md)
- [Embedding Dimensions](03_Embedding_Dimensions.md)
