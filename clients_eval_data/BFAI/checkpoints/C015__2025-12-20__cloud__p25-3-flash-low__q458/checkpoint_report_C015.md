# Checkpoint Report: C015

**Generated:** 2025-12-21 11:53:41
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

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-api-ppfq5ahfsq-ue.a.run.app |
| **Precision@K** | 25 |
| **Recall@K** | 100 |
| **Max Workers** | 100 |
| **Effective Workers** | 50 |

### Generator Model

| Parameter | Value |
|-----------|-------|
| **Model** | gemini-3-flash-preview |
| **Reasoning Effort** | low |
| **Temperature** | 0.0 |
| **Seed** | 42 |

### Judge Model

| Parameter | Value |
|-----------|-------|
| **Model** | gemini-3-flash-preview |
| **Reasoning Effort** | low |
| **Seed** | 42 |

## Retrieval Metrics

### By Question Type

| Metric | Total | Single-Hop | Multi-Hop |
|--------|-------|------------|-----------|
| **Recall@100** | 99.1% | 100.0% | 98.3% |
| **MRR** | 0.738 | 1.000 | 0.492 |

### By Difficulty

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| **Recall@100** | 100.0% | 98.8% | 98.5% |
| **MRR** | 0.752 | 0.783 | 0.669 |

## Quality Scores

### By Question Type

| Metric | Total | Single-Hop | Multi-Hop |
|--------|-------|------------|-----------|
| **Overall Score** | 4.82/5 | 4.86/5 | 4.79/5 |
| **Pass Rate** | 92.8% | 93.7% | 91.9% |

### By Difficulty

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| **Overall Score** | 4.80/5 | 4.84/5 | 4.83/5 |
| **Pass Rate** | 91.9% | 93.2% | 93.4% |

### Score Dimensions (Total)

| Dimension | Average |
|-----------|---------|
| **Correctness** | 4.78/5 |
| **Completeness** | 4.87/5 |
| **Faithfulness** | 4.90/5 |
| **Relevance** | 4.98/5 |
| **Clarity** | 4.99/5 |

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
| **Single-Hop** | 222 | 208 | 12 | 2 | 93.7% |
| **Multi-Hop** | 236 | 217 | 16 | 3 | 91.9% |

## Breakdown by Difficulty

| Difficulty | Total | Pass | Partial | Fail | Pass Rate |
|------------|-------|------|---------|------|-----------|
| **Easy** | 161 | 148 | 9 | 4 | 91.9% |
| **Medium** | 161 | 150 | 11 | 0 | 93.2% |
| **Hard** | 136 | 127 | 8 | 1 | 93.4% |

## Breakdown by Type × Difficulty

| Type | Difficulty | Count | Pass Rate | MRR | Overall Score |
|------|------------|-------|-----------|-----|---------------|
| **Single-Hop** | Easy | 88 | 95.5% | 1.000 | 4.87/5 |
| **Single-Hop** | Medium | 78 | 91.0% | 1.000 | 4.81/5 |
| **Single-Hop** | Hard | 56 | 94.6% | 1.000 | 4.90/5 |
| **Multi-Hop** | Easy | 73 | 87.7% | 0.452 | 4.71/5 |
| **Multi-Hop** | Medium | 83 | 95.2% | 0.580 | 4.86/5 |
| **Multi-Hop** | Hard | 80 | 92.5% | 0.437 | 4.79/5 |

## Execution & Throttling

| Metric | Value |
|--------|-------|
| **Run Duration** | 535.1s |
| **Questions/Second** | 0.86 |
| **Max Workers** | 100 |
| **Effective Workers** | 50 |
| **Total Requests** | 458 |
| **Total Throttles** | 0 |
| **RPM Utilization** | 0.04% |
| **TPM Utilization** | 0.04% |

## Index & Orchestrator

| Field | Value |
|-------|-------|
| **Index/Job ID** | bfai__eval66a_g1_1536_tt |
| **Mode** | cloud |
| **Endpoint** | https://bfai-api-ppfq5ahfsq-ue.a.run.app |
| **Service** | bfai-api |
| **Project ID** | bfai-prod |
| **Environment** | production |
| **Region** | us-east1 |

---

*Report generated: 2025-12-21 11:53:41*
*Checkpoint: C015*
*Judge Model: gemini-3-flash-preview*