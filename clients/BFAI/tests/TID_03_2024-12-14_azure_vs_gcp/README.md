# Azure vs GCP RAG Comparison

**Date:** December 14, 2024  
**Status:** Completed

## Objective

Compare Azure AI Search + Azure OpenAI against GCP Vertex AI for RAG quality on an apples-to-apples basis.

## Platforms Tested

| Component | Azure | GCP |
|-----------|-------|-----|
| Search | Azure AI Search | Vertex AI Vector Search |
| Embeddings | text-embedding-3-large (3072 dim) | gemini-embedding-001 (768 dim) |
| Generation | GPT-4.1-mini | Gemini 2.5 Flash |
| Judge | Gemini 2.5 Flash | Gemini 2.5 Flash |

## Key Findings

**Winner: GCP Vertex AI** by +0.23 points and +13% pass rate

| Metric | Azure | GCP | Delta |
|--------|-------|-----|-------|
| Overall Score | 3.90 | 4.13 | +0.23 |
| Pass Rate | 56% | 69% | +13% |
| Correctness | 3.53 | 4.06 | +0.53 |
| Completeness | 3.41 | 3.92 | +0.51 |
| Faithfulness | 3.65 | 3.84 | +0.19 |

### Why This Comparison Is Fair

- Same judge (Gemini 2.5 Flash) eliminates judge bias
- Same retrieval approach (top-12 vector search, no reranker)
- Same question set (193 questions, excluding 30 from docs missing in Azure)
- Both hit raw search endpoints, no orchestration layer

### Insights

1. GCP wins despite smaller embeddings (768 vs 3072 dimensions)
2. Correctness is the biggest gap (+0.53)
3. Azure is missing 8 documents from the Q&A corpus (20% of sources)

## Configuration

- **Corpus:** 193 questions (filtered from 224 to exclude missing docs)
- **Retrieval:** Top-12 chunks, pure vector search
- **Reranker:** None (apples-to-apples)

## Data Files

- `azure_evaluation_20251214_183826.json` - Final Azure results (224 questions)
- `checkpoint_azure.json` - Azure checkpoint
- `azure_evaluation_*.json` - Earlier Azure runs

## How to Reproduce

```bash
cd scripts/
python azure_comparison.py --filter-missing --workers 8
```

## See Also

- `reports/GCP_vs_Azure_RAG_Comparison.md` - Full comparison report
- `corpus/document_inventory.md` - Document overlap analysis
