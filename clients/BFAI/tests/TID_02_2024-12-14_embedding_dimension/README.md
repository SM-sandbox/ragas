# Embedding Dimension Test

**Date:** December 14, 2024  
**Status:** Completed

## Objective

Test whether larger embedding dimensions (1536) improve retrieval quality compared to the default 768 dimensions for gemini-embedding-001.

## Models Tested

| Model | Dimensions | Task Type |
|-------|------------|-----------|
| gemini-embedding-001 | 768 | RETRIEVAL_QUERY |
| gemini-embedding-001 | 1536 | RETRIEVAL_QUERY |

## Key Findings

**Result: Larger dimensions do NOT improve performance**

| Dimensions | Score | Pass% |
|------------|-------|-------|
| 768 | 4.25 | 72% |
| 1536 | 4.21 | 71% |

### Insights

1. **768 dimensions is sufficient:** The smaller embedding size performs equally well or slightly better
2. **Cost/performance tradeoff:** 768 dimensions uses less storage and compute with no quality loss
3. **Recommendation:** Stick with 768 dimensions for gemini-embedding-001

## Configuration

- **Corpus:** 224 questions from 40 source documents
- **Retrieval:** Top-12 chunks via Vertex AI Vector Search
- **Generation:** Gemini 2.5 Flash
- **Judge:** Gemini 2.5 Flash
- **Reranker:** Vertex AI Ranking API

## Data Files

- `embedding_comparison_rerank_20251214_191900.json` - 1536 dimension results with reranker
- `checkpoint_gemini-1536-RETRIEVAL_QUERY.json` - Checkpoint
- `embedding_gemini-1536-RETRIEVAL_QUERY_intermediate.json` - Intermediate results

## How to Reproduce

```bash
cd scripts/
# Requires deploying a 1536-dimension index first
python embedding_comparison_direct.py --config gemini-1536-RETRIEVAL_QUERY --mode rerank
```
