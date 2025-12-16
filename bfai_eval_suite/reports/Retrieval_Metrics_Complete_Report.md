# Complete Retrieval Metrics Report

**Generated:** 2025-12-15
**Questions Evaluated:** 160 (questions with source_document field)
**Total Corpus:** 224 questions

## Summary

| Config | Platform | Recall@10 | MRR | Notes |
|--------|----------|-----------|-----|-------|
| **gemini-RETRIEVAL_QUERY** | GCP | **90.6%** | **0.801** | 🏆 Best overall |
| text-embedding-005 | GCP | 86.9% | 0.760 | Baseline |
| gemini-SEMANTIC_SIMILARITY | GCP | 83.1% | 0.612 | Wrong task type |
| azure-text-embedding-3-large | Azure | 70.0% | 0.608 | Staging (40 docs) |

## Recall@K (All Configs)

Percentage of questions where the source document appears in top-K results.

| Config | R@5 | R@10 | R@15 | R@20 | R@25 | R@50 | R@100 |
|--------|------|------|------|------|------|------|------|
| gemini-RETRIEVAL_QUERY | 84.4% | 90.6% | 91.2% | 92.5% | 93.8% | 96.9% | 97.5% |
| text-embedding-005 | 81.9% | 86.9% | 89.4% | 91.9% | 95.6% | 96.9% | 99.4% |
| gemini-SEMANTIC_SIMILARITY | 78.1% | 83.1% | 85.6% | 86.9% | 87.5% | 90.6% | 94.4% |
| azure-text-embedding-3-large | 67.5% | 70.0% | 71.2% | 72.5% | 74.4% | 80.0% | 83.8% |

## Precision@K (All Configs)

Average percentage of top-K results that are from the correct source document.

| Config | P@5 | P@10 | P@15 | P@20 | P@25 | P@50 | P@100 |
|--------|------|------|------|------|------|------|------|
| gemini-RETRIEVAL_QUERY | 54.4% | 44.4% | 39.1% | 36.0% | 34.1% | 27.6% | 20.8% |
| text-embedding-005 | 53.0% | 42.9% | 38.5% | 35.5% | 33.3% | 27.6% | 21.5% |
| gemini-SEMANTIC_SIMILARITY | 41.9% | 33.9% | 29.6% | 27.1% | 25.1% | 20.1% | 15.1% |
| azure-text-embedding-3-large | 45.9% | 39.6% | 36.8% | 34.4% | 32.7% | 27.8% | 22.7% |

## MRR (Mean Reciprocal Rank)

Average of 1/rank of the first relevant result.

| Config | MRR | MRR@5 | MRR@10 | MRR@20 | MRR@50 | MRR@100 |
|--------|-----|--------|--------|--------|--------|--------|
| gemini-RETRIEVAL_QUERY | 0.801 | 0.790 | 0.799 | 0.800 | 0.801 | 0.801 |
| text-embedding-005 | 0.760 | 0.747 | 0.754 | 0.758 | 0.760 | 0.760 |
| gemini-SEMANTIC_SIMILARITY | 0.612 | 0.602 | 0.608 | 0.610 | 0.612 | 0.612 |
| azure-text-embedding-3-large | 0.608 | 0.600 | 0.603 | 0.605 | 0.607 | 0.608 |

## Key Findings

### 1. Best Performer: gemini-RETRIEVAL_QUERY
- **Recall@10: 90.6%** - 9 out of 10 questions find the right document in top 10
- **MRR: 0.801** - On average, the correct document appears at rank ~1.25
- Uses `RETRIEVAL_QUERY` task type which is optimized for search queries

### 2. Task Type Matters for Gemini
- `RETRIEVAL_QUERY` outperforms `SEMANTIC_SIMILARITY` by **7.5% on Recall@10**
- MRR difference: 0.801 vs 0.612 (31% improvement)
- Always use `RETRIEVAL_QUERY` for query embeddings with gemini-embedding-001

### 3. Azure Performance
- Lower recall (70% vs 90.6%) likely due to:
  - Different document set (staging with 40 docs vs production)
  - Different chunking strategy
  - No hybrid search (dense only)
- Still usable but GCP gemini-RETRIEVAL_QUERY is significantly better

### 4. text-embedding-005 is Solid Baseline
- 86.9% Recall@10 without any task type configuration
- Simpler to use (no task type needed)
- Good fallback if gemini models have issues

## Methodology

- **Recall@K**: Binary - did the correct source document appear anywhere in top-K?
- **Precision@K**: What fraction of top-K results came from the correct source?
- **MRR**: 1/rank of first relevant result, averaged across all questions
- **Search Method**: Hybrid search (dense + sparse with RRF fusion) for GCP, dense-only for Azure

## Metric Definitions

- **Recall@K**: Did the correct source document appear anywhere in the top-K results?
- **Precision@K**: What fraction of the top-K results came from the correct source?
- **MRR**: How highly was the first relevant result ranked? (1/rank, averaged)
- **MRR@K**: MRR but only considering results in top-K (0 if not in top-K)
