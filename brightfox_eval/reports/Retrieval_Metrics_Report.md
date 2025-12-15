# Retrieval Metrics Report

**Generated:** 2025-12-15 08:39:15
**Questions Evaluated:** 160

## Recall@K

Percentage of questions where the source document appears in top-K results.

| Config | R@5 | R@10 | R@15 | R@20 | R@25 | R@50 | R@100 |
|--------|------|------|------|------|------|------|------|
| azure-text-embedding-3-large | 67.5% | 70.0% | 71.2% | 72.5% | 74.4% | 80.0% | 83.8% |

## Precision@K

Average percentage of top-K results that are from the correct source document.

| Config | P@5 | P@10 | P@15 | P@20 | P@25 | P@50 | P@100 |
|--------|------|------|------|------|------|------|------|
| azure-text-embedding-3-large | 45.9% | 39.6% | 36.8% | 34.4% | 32.7% | 27.8% | 22.7% |

## MRR (Mean Reciprocal Rank)

Average of 1/rank of the first relevant result.

| Config | MRR | MRR@5 | MRR@10 | MRR@15 | MRR@20 | MRR@25 | MRR@50 | MRR@100 |
|--------|-----|--------|--------|--------|--------|--------|--------|--------|
| azure-text-embedding-3-large | 0.608 | 0.600 | 0.603 | 0.604 | 0.605 | 0.606 | 0.607 | 0.608 |

## Metric Definitions

- **Recall@K**: Did the correct source document appear anywhere in the top-K results?
- **Precision@K**: What fraction of the top-K results came from the correct source?
- **MRR**: How highly was the first relevant result ranked? (1/rank, averaged)
- **MRR@K**: MRR but only considering results in top-K (0 if not in top-K)