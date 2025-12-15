# Retrieval Metrics Report

**Generated:** 2025-12-15 00:16:45
**Questions Evaluated:** 160

## Recall@K

Percentage of questions where the source document appears in top-K results.

| Config | R@5 | R@10 | R@15 | R@20 | R@25 | R@50 | R@100 |
|--------|------|------|------|------|------|------|------|
| azure-text-embedding-3-large | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| gemini-RETRIEVAL_QUERY | 84.4% | 90.6% | 91.2% | 92.5% | 93.8% | 96.9% | 97.5% |
| gemini-SEMANTIC_SIMILARITY | 78.1% | 83.1% | 85.6% | 86.9% | 87.5% | 90.6% | 94.4% |
| text-embedding-005 | 81.9% | 86.9% | 89.4% | 91.9% | 95.6% | 96.9% | 99.4% |

## Precision@K

Average percentage of top-K results that are from the correct source document.

| Config | P@5 | P@10 | P@15 | P@20 | P@25 | P@50 | P@100 |
|--------|------|------|------|------|------|------|------|
| azure-text-embedding-3-large | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| gemini-RETRIEVAL_QUERY | 54.4% | 44.4% | 39.1% | 36.0% | 34.1% | 27.6% | 20.8% |
| gemini-SEMANTIC_SIMILARITY | 41.9% | 33.9% | 29.6% | 27.1% | 25.1% | 20.1% | 15.1% |
| text-embedding-005 | 53.0% | 42.9% | 38.5% | 35.5% | 33.3% | 27.6% | 21.5% |

## MRR (Mean Reciprocal Rank)

Average of 1/rank of the first relevant result.

| Config | MRR | MRR@5 | MRR@10 | MRR@15 | MRR@20 | MRR@25 | MRR@50 | MRR@100 |
|--------|-----|--------|--------|--------|--------|--------|--------|--------|
| azure-text-embedding-3-large | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gemini-RETRIEVAL_QUERY | 0.801 | 0.790 | 0.799 | 0.799 | 0.800 | 0.800 | 0.801 | 0.801 |
| gemini-SEMANTIC_SIMILARITY | 0.612 | 0.602 | 0.608 | 0.610 | 0.610 | 0.611 | 0.612 | 0.612 |
| text-embedding-005 | 0.760 | 0.747 | 0.754 | 0.756 | 0.758 | 0.759 | 0.760 | 0.760 |

## Metric Definitions

- **Recall@K**: Did the correct source document appear anywhere in the top-K results?
- **Precision@K**: What fraction of the top-K results came from the correct source?
- **MRR**: How highly was the first relevant result ranked? (1/rank, averaged)
- **MRR@K**: MRR but only considering results in top-K (0 if not in top-K)