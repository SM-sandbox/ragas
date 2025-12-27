# Checkpoint Report: C069

**Generated:** 2025-12-26 15:09:30
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 88.0% |
| **Partial Rate** | 8.5% |
| **Fail Rate** | 3.5% |
| **Acceptable Rate** | 96.5% |
| **Overall Score** | 4.71/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 88.0% | -4.1% | ⚠️ |
| **Acceptable Rate** | 98.9% | 96.5% | -2.4% | ❌ |
| **Fail Rate** | 1.1% | 3.5% | +2.4% | ❌ |
| **MRR** | 0.740 | 0.740 | -0.000 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.71/5 | -0.11 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 73.8s | +16.1s | 0.78x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.9612 | $+0.0949 |
| **Per Question** | $0.001892 | $0.002099 | $+0.000207 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 89.6% | -3.2% |
| **Multi-Hop** | 91.5% | 86.4% | -5.1% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 82.6% | -6.8% |
| **Medium** | 93.2% | 91.9% | -1.2% |
| **Hard** | 94.1% | 89.7% | -4.4% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 4.1%, acceptable rate dropped 2.4%, latency increased 16.1s

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
| **Model** | gemini-2.5-flash |
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
|**MRR**         |  0.740|       1.000|      0.494|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.786|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.71/5|      4.77/5|     4.66/5|
|**Pass Rate**      |   88.0%|       89.6%|      86.4%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.60/5|  4.81/5|  4.73/5|
|**Pass Rate**      |   82.6%|   91.9%|   89.7%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.70/5|  4.63/5|  4.78/5|  4.68/5|
|**Completeness**  |  4.79/5|  4.71/5|  4.86/5|  4.81/5|
|**Faithfulness**  |  4.74/5|  4.58/5|  4.80/5|  4.85/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.96/5|  4.97/5|
|**Clarity**       |  4.98/5|  4.97/5|  4.98/5|  4.99/5|
|**Overall**       |  4.71/5|  4.60/5|  4.81/5|  4.73/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.70/5|      4.77/5|     4.63/5|
|**Completeness**  |  4.79/5|      4.85/5|     4.74/5|
|**Faithfulness**  |  4.74/5|      4.68/5|     4.79/5|
|**Relevance**     |  4.96/5|      4.94/5|     4.97/5|
|**Clarity**       |  4.98/5|      4.97/5|     4.99/5|
|**Overall**       |  4.71/5|      4.77/5|     4.66/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.16s|        2.9%|
|**Total**       |    73.78s|        100%|

**Min Latency:** 13.00s  |  **Max Latency:** 140.77s

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
|**Single-Hop**  |    222|   199|       13|    10|      89.6%|
|**Multi-Hop**   |    236|   204|       26|     6|      86.4%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   133|       19|     9|      82.6%|
|**Medium**  |    161|   148|        9|     4|      91.9%|
|**Hard**    |    136|   122|       11|     3|      89.7%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      85.2%|  1.000|         4.69/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.83/5|
|**Single-Hop**  |        Hard|     56|      92.9%|  1.000|         4.80/5|
|**Multi-Hop**   |        Easy|     73|      79.5%|  0.452|         4.51/5|
|**Multi-Hop**   |      Medium|     83|      91.6%|  0.586|         4.79/5|
|**Multi-Hop**   |        Hard|     80|      87.5%|  0.438|         4.68/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  425.7s|
|**Questions/Second**   |    1.08|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.29%|
|**TPM Utilization**    |   0.29%|

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

*Report generated: 2025-12-26 15:09:30*
*Checkpoint: C069*
*Judge Model: gemini-3-flash-preview*