# LLM Judge Baseline Report

**Date:** 2024-12-13  
**Experiment:** `experiments/2024-12-14_embedding_model_comparison/data/llm_judge_results.json`

---

## 1. Overview

This report documents the baseline LLM-as-Judge evaluation results from December 13, 2024. These results serve as the comparison baseline for subsequent E2E orchestrator tests.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Total Questions | 224 |
| Single-Hop Questions | 160 |
| Multi-Hop Questions | 64 |
| Embedding | gemini-embedding-001 (768 dim) |
| Task Type | RETRIEVAL_QUERY |
| Generation Model | Gemini 2.5 Flash |
| Judge Model | Gemini 2.5 Flash |
| Retrieval | Top-12 chunks |
| Reranker | Vertex AI Ranking API |

---

## 3. Overall Results

| Metric | Score |
|--------|-------|
| **Overall Score** | 4.16/5 |
| **Pass Rate** | 82.6% |
| Correctness | 4.13/5 |
| Completeness | 3.98/5 |
| Faithfulness | 4.95/5 |
| Relevance | 4.67/5 |
| Clarity | 4.97/5 |

### Verdict Distribution

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 151 | 67.4% |
| Partial | 34 | 15.2% |
| Fail | 39 | 17.4% |
| Error | 0 | 0% |

---

## 4. Single-Hop vs Multi-Hop Breakdown

| Metric | Single-Hop (160) | Multi-Hop (64) | Delta |
|--------|------------------|----------------|-------|
| Overall Score | 4.26 | 3.91 | -0.35 |
| Correctness | 4.22 | 3.91 | -0.31 |
| Completeness | 4.18 | 3.47 | -0.71 |
| Faithfulness | 4.93 | 4.98 | +0.05 |
| Relevance | 4.64 | 4.77 | +0.13 |
| Clarity | 4.97 | 4.98 | +0.01 |

### Key Insight

**Multi-hop questions are significantly harder:**
- Overall score drops by 0.35 points
- Completeness drops by 0.71 points (biggest gap)
- Multi-hop questions require synthesizing information from multiple sources

---

## 5. Comparison with E2E Orchestrator Test (2025-12-15)

| Metric | Baseline (Dec 13) | E2E Test (Dec 15) | Improvement |
|--------|-------------------|-------------------|-------------|
| Overall Score | 4.16/5 | 4.43/5 | **+0.27** |
| Pass Rate | 82.6% | 89.0% | **+6.4%** |
| Correctness | 4.13/5 | 4.40/5 | +0.27 |
| Faithfulness | 4.95/5 | 4.98/5 | +0.03 |

### What Changed?

| Factor | Baseline | E2E Test |
|--------|----------|----------|
| Embedding Dimensions | 768 | 1536 |
| Search Type | Pure vector | Hybrid 50/50 |
| Retrieval Pool | Top 12 | Top 100 → rerank to 12 |
| Judge Model | Gemini 2.5 Flash | Gemini 2.0 Flash |
| Judge Temperature | Unknown | 0.0 |

---

## 6. Failure Analysis

### Common Failure Patterns (from 39 failed questions)

1. **Multi-hop synthesis** - Questions requiring information from multiple documents
2. **Specific numerical values** - Exact specifications that weren't in retrieved chunks
3. **Ambiguous source documents** - Questions where the expected source wasn't clear

---

## 7. Conclusions

1. **Strong baseline performance** - 82.6% pass rate with 4.16/5 overall score

2. **Multi-hop is the challenge** - 0.35 point drop for multi-hop questions

3. **Faithfulness is excellent** - 4.95/5 means the model rarely hallucinates

4. **Room for improvement** - Completeness (3.98) is the weakest dimension

5. **Subsequent improvements** - E2E test achieved +6.4% pass rate through hybrid search and better reranking

---

## 8. Data Files

- `experiments/2024-12-14_embedding_model_comparison/data/llm_judge_results.json` (293KB)
- Contains detailed per-question judgments with explanations
