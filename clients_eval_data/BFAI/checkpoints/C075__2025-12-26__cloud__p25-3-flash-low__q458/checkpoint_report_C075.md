# Checkpoint Report: C075

**Generated:** 2025-12-26 21:54:33
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 93.0% |
| **Partial Rate** | 5.5% |
| **Fail Rate** | 1.5% |
| **Acceptable Rate** | 98.5% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 93.0% | +0.9% | ✅ |
| **Acceptable Rate** | 98.9% | 98.5% | -0.4% | ✅ |
| **Fail Rate** | 1.1% | 1.5% | +0.4% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.82/5 | +0.00 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 49.3s | -8.4s | 1.17x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.8438 | $-0.0226 |
| **Per Question** | $0.001892 | $0.001842 | $-0.000049 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 95.5% | +2.7% |
| **Multi-Hop** | 91.5% | 90.7% | -0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 89.4% | +0.0% |
| **Medium** | 93.2% | 96.9% | +3.7% |
| **Hard** | 94.1% | 92.6% | -1.5% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-eb2qyzdzvq-ue.a.run.app |
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
|**MRR**         |  0.739|       1.000|      0.493|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.780|  0.671|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.86/5|     4.79/5|
|**Pass Rate**      |   93.0%|       95.5%|      90.7%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.74/5|  4.89/5|  4.83/5|
|**Pass Rate**      |   89.4%|   96.9%|   92.6%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.78/5|  4.69/5|  4.88/5|  4.77/5|
|**Completeness**  |  4.88/5|  4.80/5|  4.97/5|  4.85/5|
|**Faithfulness**  |  4.86/5|  4.77/5|  4.92/5|  4.91/5|
|**Relevance**     |  4.97/5|  4.92/5|  4.99/5|  4.99/5|
|**Clarity**       |  5.00/5|  4.99/5|  5.00/5|  5.00/5|
|**Overall**       |  4.82/5|  4.74/5|  4.89/5|  4.83/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.78/5|      4.86/5|     4.71/5|
|**Completeness**  |  4.88/5|      4.92/5|     4.83/5|
|**Faithfulness**  |  4.86/5|      4.81/5|     4.92/5|
|**Relevance**     |  4.97/5|      4.96/5|     4.97/5|
|**Clarity**       |  5.00/5|      5.00/5|     5.00/5|
|**Overall**       |  4.82/5|      4.86/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.92s|        3.9%|
|**Total**       |    49.30s|        100%|

**Min Latency:** 8.72s  |  **Max Latency:** 96.85s

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
|**Single-Hop**  |    222|   212|        5|     5|      95.5%|
|**Multi-Hop**   |    236|   214|       20|     2|      90.7%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       11|     6|      89.4%|
|**Medium**  |    161|   156|        5|     0|      96.9%|
|**Hard**    |    136|   126|        9|     1|      92.6%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.79/5|
|**Single-Hop**  |      Medium|     78|      98.7%|  1.000|         4.91/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.89/5|
|**Multi-Hop**   |        Easy|     73|      86.3%|  0.459|         4.67/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.574|         4.88/5|
|**Multi-Hop**   |        Hard|     80|      90.0%|  0.440|         4.79/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  289.4s|
|**Questions/Second**   |    1.58|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.34%|
|**TPM Utilization**    |   0.34%|

## Index & Orchestrator

|Field             |                                     Value|
|------------------|-----------------------------------------:|
|**Index/Job ID**  |                  bfai__eval66a_g1_1536_tt|
|**Mode**          |                                     cloud|
|**Endpoint**      |  https://bfai-app-eb2qyzdzvq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 21:54:33*
*Checkpoint: C075*
*Judge Model: gemini-3-flash-preview*