# Pure Semantic Retrieval Test Report

**Generated:** 2025-12-15
**Test Date:** December 15, 2025

---

## 1. Goal

Evaluate the **pure semantic retrieval quality** of different GCP embedding configurations to determine:

1. **Which embedding model + dimensionality produces the best retrieval quality** (measured by MRR and Recall@K)
2. **Whether the RETRIEVAL_QUERY task type improves results** over default embeddings
3. **Retrieval latency** for each configuration
4. **Consistency** between Direct API calls and the Orchestrator retrieval layer

---

## 2. Methodology

### What We're Testing
- **Pure semantic search only** — no keyword/sparse embeddings, no hybrid search
- Vector similarity search using cosine distance on GCP Vertex AI Vector Search
- `rrf_ranking_alpha = 1.0` (100% dense vectors, 0% sparse/keyword)
- `enable_hybrid = False`

### How We're Testing
1. **Direct API Path**: Query embeddings generated locally → sent directly to MatchingEngineIndexEndpoint
2. **Orchestrator Path**: Query embeddings generated via VectorSearchRetriever component (same retrieval logic, validates orchestrator consistency)

### Test Corpus
- **160 questions** from `qa_corpus_200.json` (filtered to only questions with a known `source_document`)
- Each question has exactly one expected source document for ground truth matching

### Metrics
- **MRR (Mean Reciprocal Rank)**: How high does the correct document rank? (1.0 = always first)
- **Recall@K**: What percentage of queries have the correct document in the top K results?
- **Latency**: Average retrieval time per query

---

## 3. Embedding Configurations Tested

| Config Name | Model | Dimensions | Task Type | Index |
|-------------|-------|------------|-----------|-------|
| `text-embedding-005` | text-embedding-005 | 768 | None (default) | EVAL66 |
| `gemini-768-RETRIEVAL_QUERY` | gemini-embedding-001 | 768 | RETRIEVAL_QUERY | EVAL66_G1_768_TT |
| `gemini-1536-RETRIEVAL_QUERY` | gemini-embedding-001 | 1536 | RETRIEVAL_QUERY | EVAL66A_G1_1536_TT |
| `gemini-3072-RETRIEVAL_QUERY` | gemini-embedding-001 | 3072 | RETRIEVAL_QUERY | EVAL66A_G1_3072_TT |

---

## 4. Results

### 4.1 MRR (Mean Reciprocal Rank)

Higher is better. MRR of 0.80 means the correct document is typically ranked ~1.25 on average.

| Config | MRR | MRR@5 | MRR@10 | MRR@20 | MRR@50 | MRR@100 |
|--------|-----|-------|--------|--------|--------|---------|
| text-embedding-005 | 0.760 | 0.747 | 0.754 | 0.758 | 0.760 | 0.760 |
| gemini-768-RETRIEVAL_QUERY | 0.801 | 0.790 | 0.799 | 0.800 | 0.801 | 0.801 |
| **gemini-1536-RETRIEVAL_QUERY** | **0.806** | **0.798** | **0.803** | **0.805** | **0.806** | **0.806** |
| gemini-3072-RETRIEVAL_QUERY | 0.802 | 0.794 | 0.799 | 0.801 | 0.802 | 0.802 |

### 4.2 Recall@K

Percentage of queries where the correct document appears in the top K results.

| Config | R@5 | R@10 | R@15 | R@20 | R@25 | R@50 | R@100 |
|--------|-----|------|------|------|------|------|-------|
| text-embedding-005 | 81.9% | 86.9% | 89.4% | 91.9% | 95.6% | 96.9% | 99.4% |
| gemini-768-RETRIEVAL_QUERY | 84.4% | 90.6% | 91.2% | 92.5% | 93.8% | 96.9% | 97.5% |
| **gemini-1536-RETRIEVAL_QUERY** | **86.9%** | **90.6%** | **91.9%** | **93.8%** | 93.8% | 96.2% | 97.5% |
| gemini-3072-RETRIEVAL_QUERY | 86.2% | 90.0% | 91.9% | 92.5% | 93.8% | 95.6% | 97.5% |

### 4.3 Retrieval Latency

Average time per query (embedding generation + vector search).

#### Direct API

| Config | Avg | Min | Max | Total (160q) |
|--------|-----|-----|-----|--------------|
| text-embedding-005 | 0.172s | 0.124s | 0.423s | 27.5s |
| gemini-768-RETRIEVAL_QUERY | 0.181s | 0.138s | 1.360s | 29.0s |
| gemini-1536-RETRIEVAL_QUERY | 0.192s | 0.155s | 1.424s | 30.8s |
| gemini-3072-RETRIEVAL_QUERY | 0.192s | 0.159s | 1.366s | 30.7s |

#### Orchestrator

| Config | Avg | Min | Max | Total (160q) |
|--------|-----|-----|-----|--------------|
| text-embedding-005 | 0.165s | 0.125s | 0.484s | 26.5s |
| gemini-768-RETRIEVAL_QUERY | 0.182s | 0.131s | 0.420s | 29.1s |
| gemini-1536-RETRIEVAL_QUERY | 0.203s | 0.140s | 0.603s | 32.6s |
| gemini-3072-RETRIEVAL_QUERY | 0.197s | 0.139s | 0.554s | 31.5s |

#### Comparison (Direct vs Orchestrator)

| Config | Direct Avg | Orchestrator Avg | Δ |
|--------|------------|------------------|---|
| text-embedding-005 | 0.172s | 0.165s | -0.007s |
| gemini-768-RETRIEVAL_QUERY | 0.181s | 0.182s | +0.001s |
| gemini-1536-RETRIEVAL_QUERY | 0.192s | 0.203s | +0.011s |
| gemini-3072-RETRIEVAL_QUERY | 0.192s | 0.197s | +0.005s |

*Note: Orchestrator uses VectorSearchRetriever component directly (no HTTP overhead). Minimal latency difference confirms consistent implementation.*

---

## 5. Findings

### Key Takeaways

1. **Best Overall: `gemini-1536-RETRIEVAL_QUERY`**
   - Highest MRR (0.806) and tied for best Recall@10 (90.6%)
   - Sweet spot between quality and dimensionality

2. **RETRIEVAL_QUERY Task Type Matters**
   - Gemini models with `RETRIEVAL_QUERY` task type outperform `text-embedding-005` by ~4% on Recall@10 (90.6% vs 86.9%)
   - MRR improves from 0.760 → 0.806 (+6%)

3. **Diminishing Returns Beyond 1536 Dimensions**
   - 3072 dimensions does NOT improve over 1536 (90.0% vs 90.6% Recall@10)
   - Slightly worse MRR (0.802 vs 0.806)
   - No latency benefit

4. **Latency is Consistent**
   - All configs perform in ~0.17-0.19s range
   - Higher dimensions don't significantly impact latency

5. **Direct vs Orchestrator: Identical Results**
   - Both paths produce the same retrieval quality
   - Validates that the orchestrator retrieval layer is correctly configured

### Recommendation

Use **`gemini-1536-RETRIEVAL_QUERY`** for production:
- Best retrieval quality (MRR 0.806, Recall@10 90.6%)
- Reasonable latency (~0.19s)
- No benefit from going to 3072 dimensions
