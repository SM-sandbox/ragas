# Checkpoint Report: C031

**Generated:** 2025-12-23 08:05:39
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.9% |
| **Partial Rate** | 5.9% |
| **Fail Rate** | 2.2% |
| **Acceptable Rate** | 97.8% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 91.9% | -0.9% | ✅ |
| **Acceptable Rate** | 98.5% | 97.8% | -0.7% | ✅ |
| **Fail Rate** | 1.5% | 2.2% | +0.7% | ✅ |
| **MRR** | 0.737 | 0.744 | +0.006 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.81/5 | -0.02 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 66.2s | +23.5s | 0.65x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8640 | $+0.0206 |
| **Per Question** | $0.001841 | $0.001886 | $+0.000045 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 94.6% | -0.9% |
| **Multi-Hop** | 90.3% | 89.4% | -0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 89.4% | -1.2% |
| **Medium** | 95.0% | 95.0% | +0.0% |
| **Hard** | 92.6% | 91.2% | -1.5% |

### Key Finding

⚠️ **Regressions detected:** latency increased 23.5s

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | local |
| **Endpoint** | N/A |
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

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.744|       1.000|      0.503|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.766|   0.782|  0.672|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.83/5|     4.78/5|
|**Pass Rate**      |   91.9%|       94.6%|      89.4%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.73/5|  4.89/5|  4.80/5|
|**Pass Rate**      |   89.4%|   95.0%|   91.2%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.70/5|  4.85/5|  4.72/5|
|**Completeness**  |  4.87/5|  4.83/5|  4.94/5|  4.82/5|
|**Faithfulness**  |  4.93/5|  4.92/5|  4.96/5|  4.93/5|
|**Relevance**     |  4.95/5|  4.92/5|  4.99/5|  4.93/5|
|**Clarity**       |  4.98/5|  4.99/5|  4.99/5|  4.97/5|
|**Overall**       |  4.81/5|  4.73/5|  4.89/5|  4.80/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.81/5|     4.71/5|
|**Completeness**  |  4.87/5|      4.91/5|     4.83/5|
|**Faithfulness**  |  4.93/5|      4.95/5|     4.92/5|
|**Relevance**     |  4.95/5|      4.93/5|     4.96/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.81/5|      4.83/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.72s|        1.1%|
|**Reranking**   |     4.23s|        6.4%|
|**Generation**  |    59.72s|       90.2%|
|**Judge**       |     1.57s|        2.4%|
|**Total**       |    66.24s|        100%|

**Min Latency:** 24.36s  |  **Max Latency:** 136.15s

## Token & Cost Analysis

### Token Breakdown

|Token Type               |  Total|  Per Question|
|-------------------------|------:|-------------:|
|**Prompt (Input)**       |      0|             0|
|**Completion (Output)**  |      0|             0|
|**Thinking**             |      0|             0|
|**Cached**               |      0|             0|
|**Total**                |      0|             0|

### Cost Estimate (Gemini 3 Flash)

|Component                |         Cost|
|-------------------------|------------:|
|**Input**                |      $0.0000|
|**Output**               |      $0.0000|
|**Thinking**             |      $0.0000|
|**Cached Savings**       |     -$0.0000|
|**Total**                |  **$0.0000**|
|**Per Question**         |    $0.000000|
|**Per 1,000 Questions**  |        $0.00|

## Breakdown by Question Type

|Type            |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|----------------|------:|-----:|--------:|-----:|----------:|
|**Single-Hop**  |    222|   210|        5|     7|      94.6%|
|**Multi-Hop**   |    236|   211|       22|     3|      89.4%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       10|     7|      89.4%|
|**Medium**  |    161|   153|        8|     0|      95.0%|
|**Hard**    |    136|   124|        9|     3|      91.2%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      93.2%|  1.000|         4.76/5|
|**Single-Hop**  |      Medium|     78|      96.2%|  1.000|         4.90/5|
|**Single-Hop**  |        Hard|     56|      94.6%|  1.000|         4.84/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.484|         4.69/5|
|**Multi-Hop**   |      Medium|     83|      94.0%|  0.577|         4.88/5|
|**Multi-Hop**   |        Hard|     80|      88.8%|  0.443|         4.77/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  368.1s|
|**Questions/Second**   |    1.24|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.26%|
|**TPM Utilization**    |   0.26%|

## Index & Orchestrator

|Field             |                  Value|
|------------------|----------------------:|
|**Index/Job ID**  |  bfai__eval66b_g1536tt|
|**Mode**          |                  local|
|**Endpoint**      |                    N/A|

---

*Report generated: 2025-12-23 08:05:39*
*Checkpoint: C031*
*Judge Model: gemini-3-flash-preview*