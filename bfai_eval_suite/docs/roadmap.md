# Roadmap

Future improvements and features for the BrightFox RAG Evaluation suite.

## Planned Features

### 1. Retrieval Quality Metrics (Priority: High)

Add pure retrieval metrics that measure document retrieval quality independent of generation:

| Metric | Description |
|--------|-------------|
| **Recall@K** | % of relevant docs retrieved in top-K results |
| **Precision@K** | % of top-K results that are relevant |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank of first relevant result |
| **MRR@5** | MRR considering only top-5 results |
| **MRR@10** | MRR considering only top-10 results |

These metrics focus purely on retrieval - did we bring back the right documents? - without involving generation quality.

**Implementation:**
- Compare retrieved chunk source documents against ground truth source document
- Calculate metrics at K=5, 10, 12, 20
- Add to evaluation output alongside generation metrics

### 2. Question Relevance Filtering

Filter the Q&A corpus to remove low-value questions:
- Document metadata questions (revision numbers, addresses)
- Watermark/footer content
- Non-domain-specific questions

**Status:** In Progress (question_relevance_evaluator.py)

### 3. Expanded Embedding Comparison

Test additional embedding configurations:
- gemini-embedding-001 at 3072 dimensions
- Hybrid search (vector + keyword)
- Different chunking strategies

### 4. Azure Reranker Comparison

Add Azure Semantic Ranker to match GCP reranker comparison.

### 5. Cost Analysis

Track and report API costs per evaluation run:
- Embedding API calls
- LLM generation tokens
- Judge LLM tokens

---

## Completed

- [x] Embedding model comparison (gemini vs text-embedding-005)
- [x] Embedding dimension test (768 vs 1536)
- [x] Azure vs GCP platform comparison
- [x] MkDocs documentation site
- [x] Project reorganization
- [x] **Retrieval Quality Metrics** - Recall@K, Precision@K, MRR at K=5,10,15,20,25,50,100
- [x] Question Relevance Filtering (224 questions evaluated)
- [x] genai.Client() caching fix (3.9x speedup for gemini embeddings)
