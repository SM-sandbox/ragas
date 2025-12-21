# Checkpoint Report: C015

**Generated:** 2025-12-21 11:06:08
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.8% |
| **Partial Rate** | 6.1% |
| **Fail Rate** | 1.1% |
| **Acceptable Rate** | 98.9% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 92.8% | +0.0% | ✅ |
| **Acceptable Rate** | 98.5% | 98.9% | +0.4% | ✅ |
| **Fail Rate** | 1.5% | 1.1% | -0.4% | ✅ |
| **MRR** | 0.737 | 0.738 | +0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.82/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 77.2s | +34.5s | 0.55x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8577 | $+0.0143 |
| **Per Question** | $0.001841 | $0.001873 | $+0.000031 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 93.7% | -1.8% |
| **Multi-Hop** | 90.3% | 91.9% | +1.7% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 91.9% | +1.2% |
| **Medium** | 95.0% | 93.2% | -1.9% |
| **Hard** | 92.6% | 93.4% | +0.7% |

### Key Finding

⚠️ **Regressions detected:** latency increased 34.5s

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
| **Correctness** | 4.78/5 |
| **Completeness** | 4.87/5 |
| **Faithfulness** | 4.90/5 |
| **Relevance** | 4.98/5 |
| **Clarity** | 4.99/5 |
| **Overall** | 4.82/5 |

## Latency Analysis

| Phase | Avg Time | % of Total |
|-------|----------|------------|
| **Retrieval** | 0.00s | 0.0% |
| **Reranking** | 0.00s | 0.0% |
| **Generation** | 0.00s | 0.0% |
| **Judge** | 1.26s | 1.6% |
| **Total** | 77.20s | 100% |

**Min Latency:** 8.07s
**Max Latency:** 134.54s

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
| **Mode** | cloud |
| **Documents** | 0 |
| **Embedding Model** | unknown |
| **Embedding Dimension** | 0 |

---

*Report generated: 2025-12-21 11:06:08*
*Checkpoint: C015*
*Judge Model: gemini-3-flash-preview*