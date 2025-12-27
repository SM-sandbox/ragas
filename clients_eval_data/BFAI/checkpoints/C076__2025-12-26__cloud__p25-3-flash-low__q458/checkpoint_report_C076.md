# Checkpoint Report: C076

**Generated:** 2025-12-26 23:12:45
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 93.0% |
| **Partial Rate** | 5.7% |
| **Fail Rate** | 1.3% |
| **Acceptable Rate** | 98.7% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 93.0% | +0.9% | ✅ |
| **Acceptable Rate** | 98.9% | 98.7% | -0.2% | ✅ |
| **Fail Rate** | 1.1% | 1.3% | +0.2% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.82/5 | +0.00 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 51.6s | -6.1s | 1.12x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.8450 | $-0.0214 |
| **Per Question** | $0.001892 | $0.001845 | $-0.000047 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 95.0% | +2.3% |
| **Multi-Hop** | 91.5% | 91.1% | -0.4% |

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
|**MRR**         |  0.739|       1.000|      0.494|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.781|  0.671|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.85/5|     4.80/5|
|**Pass Rate**      |   93.0%|       95.0%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.75/5|  4.89/5|  4.83/5|
|**Pass Rate**      |   89.4%|   96.9%|   92.6%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.79/5|  4.71/5|  4.88/5|  4.77/5|
|**Completeness**  |  4.88/5|  4.81/5|  4.97/5|  4.85/5|
|**Faithfulness**  |  4.87/5|  4.78/5|  4.91/5|  4.92/5|
|**Relevance**     |  4.97/5|  4.93/5|  4.99/5|  4.99/5|
|**Clarity**       |  5.00/5|  5.00/5|  5.00/5|  5.00/5|
|**Overall**       |  4.82/5|  4.75/5|  4.89/5|  4.83/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.79/5|      4.85/5|     4.72/5|
|**Completeness**  |  4.88/5|      4.92/5|     4.83/5|
|**Faithfulness**  |  4.87/5|      4.81/5|     4.93/5|
|**Relevance**     |  4.97/5|      4.96/5|     4.97/5|
|**Clarity**       |  5.00/5|      5.00/5|     5.00/5|
|**Overall**       |  4.82/5|      4.85/5|     4.80/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.84s|        3.6%|
|**Total**       |    51.59s|        100%|

**Min Latency:** 10.26s  |  **Max Latency:** 83.10s

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
|**Single-Hop**  |    222|   211|        6|     5|      95.0%|
|**Multi-Hop**   |    236|   215|       20|     1|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       12|     5|      89.4%|
|**Medium**  |    161|   156|        5|     0|      96.9%|
|**Hard**    |    136|   126|        9|     1|      92.6%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      90.9%|  1.000|         4.78/5|
|**Single-Hop**  |      Medium|     78|      98.7%|  1.000|         4.90/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.89/5|
|**Multi-Hop**   |        Easy|     73|      87.7%|  0.459|         4.70/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.576|         4.88/5|
|**Multi-Hop**   |        Hard|     80|      90.0%|  0.440|         4.79/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  298.8s|
|**Questions/Second**   |    1.53|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.41%|
|**TPM Utilization**    |   0.41%|

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

*Report generated: 2025-12-26 23:12:45*
*Checkpoint: C076*
*Judge Model: gemini-3-flash-preview*