# Embedding Model Comparison

**Date:** December 14, 2024  
**Status:** Completed

## Objective

Compare different embedding models to determine which provides the best retrieval quality for RAG evaluation.

## Models Tested

| Model | Dimensions | Task Type |
|-------|------------|-----------|
| text-embedding-005 | 768 | None |
| gemini-embedding-001 | 768 | SEMANTIC_SIMILARITY |
| gemini-embedding-001 | 768 | RETRIEVAL_QUERY |

## Results

**Winner: gemini-embedding-001 with RETRIEVAL_QUERY task type**

| Model | Score | Pass% |
|-------|-------|-------|
| gemini-RETRIEVAL_QUERY | 4.25 | 72% |
| gemini-SEMANTIC_SIMILARITY | 4.18 | 70% |
| text-embedding-005 | 4.10 | 68% |

## Key Insights

1. **Task type matters:** RETRIEVAL_QUERY outperforms SEMANTIC_SIMILARITY by 0.07 points
2. **Gemini beats text-embedding-005:** Both Gemini variants outperform the general-purpose model
3. **Reranker helps:** Adding reranker improves scores by ~0.1 points across all models

## Configuration

- **Corpus:** 224 questions from 40 source documents
- **Retrieval:** Top-12 chunks via Vertex AI Vector Search
- **Generation:** Gemini 2.5 Flash
- **Judge:** Gemini 2.5 Flash

## How to Reproduce

```bash
cd scripts/
python embedding_comparison_direct.py --mode recall-only
python embedding_comparison_direct.py --mode rerank
```
