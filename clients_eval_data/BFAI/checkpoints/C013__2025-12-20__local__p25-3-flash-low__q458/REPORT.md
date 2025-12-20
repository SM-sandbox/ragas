# Checkpoint Report: C013

**Generated:** 2025-12-20 10:46:43
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.9% |
| **Partial Rate** | 6.6% |
| **Fail Rate** | 1.5% |
| **Acceptable Rate** | 98.5% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

### Comparison to Gold Baseline (C012)

| Metric | Gold Baseline | This Run | Delta |
|--------|---------------|----------|-------|
| **Pass Rate** | 92.8% | 91.9% | -0.9% ✓ |
| **Acceptable Rate** | 98.5% | 98.5% | -0.0% ✓ |

## Configuration

| Parameter | Value |
|-----------|-------|
| **Generator Model** | gemini-3-flash-preview |
| **Reasoning Effort** | low |
| **Temperature** | 0.0 |
| **Precision@K** | 25 |
| **Recall@K** | 100 |
| **Workers** | 100 |
| **Judge Model** | gemini-3-flash-preview |

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| **Recall@100** | 99.1% |
| **MRR** | 0.738 |

## Quality Scores

| Dimension | Average |
|-----------|---------|
| **Correctness** | 4.76/5 |
| **Completeness** | 4.86/5 |
| **Faithfulness** | 4.95/5 |
| **Relevance** | 4.96/5 |
| **Clarity** | 4.98/5 |
| **Overall** | 4.82/5 |

## Latency Analysis

| Phase | Avg Time | % of Total |
|-------|----------|------------|
| **Retrieval** | 0.51s | 0.6% |
| **Reranking** | 3.13s | 3.9% |
| **Generation** | 74.78s | 94.0% |
| **Judge** | 1.10s | 1.4% |
| **Total** | 79.51s | 100% |

**Min Latency:** 13.62s
**Max Latency:** 151.72s

## Token & Cost Analysis

### Token Breakdown

| Token Type | Total | Per Question |
|------------|-------|--------------|
| **Prompt (Input)** | 0 | 0 |
| **Completion (Output)** | 0 | 0 |
| **Thinking** | 0 | 0 |
| **Cached** | 0 | 0 |
| **Total** | 0 | 0 |

### Cost Estimate (Gemini 3 Flash)

| Component | Cost |
|-----------|------|
| **Input** | $0.0000 |
| **Output** | $0.0000 |
| **Thinking** | $0.0000 |
| **Cached Savings** | -$0.0000 |
| **Total** | **$0.0000** |
| **Per Question** | $0.000000 |
| **Per 1,000 Questions** | $0.00 |

## Breakdown by Question Type

| Type | Total | Pass | Partial | Fail | Pass Rate |
|------|-------|------|---------|------|-----------|
| **Multi-Hop** | 458 | 0 | 0 | 0 | 0.0% |

## Breakdown by Difficulty

| Difficulty | Total | Pass | Partial | Fail | Pass Rate |
|------------|-------|------|---------|------|-----------|
| **Easy** | 161 | 0 | 0 | 0 | 0.0% |
| **Medium** | 161 | 0 | 0 | 0 | 0.0% |
| **Hard** | 136 | 0 | 0 | 0 | 0.0% |

## Index Information

| Field | Value |
|-------|-------|
| **Job ID** | bfai__eval66a_g1_1536_tt |
| **Mode** | local |
| **Documents** | 64 |
| **Embedding Model** | gemini-embedding-001 |
| **Embedding Dimension** | 1536 |

---

*Report generated: 2025-12-20 10:46:43*
*Checkpoint: C013*
*Judge Model: gemini-3-flash-preview*