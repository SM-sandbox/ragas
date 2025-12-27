# Checkpoint Report: C064

**Generated:** 2025-12-26 13:59:49
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.7% |
| **Partial Rate** | 6.3% |
| **Fail Rate** | 2.0% |
| **Acceptable Rate** | 98.0% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 91.7% | -0.4% | ✅ |
| **Acceptable Rate** | 98.9% | 98.0% | -0.9% | ✅ |
| **Fail Rate** | 1.1% | 2.0% | +0.9% | ✅ |
| **MRR** | 0.740 | 0.737 | -0.003 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.81/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 55.0s | -2.7s | 1.05x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.8831 | $+0.0168 |
| **Per Question** | $0.001892 | $0.001928 | $+0.000037 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 92.3% | -0.5% |
| **Multi-Hop** | 91.5% | 91.1% | -0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 88.2% | -1.2% |
| **Medium** | 93.2% | 93.2% | +0.0% |
| **Hard** | 94.1% | 94.1% | +0.0% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-xcshqh7sqq-ue.a.run.app |
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
|**MRR**         |  0.737|       1.000|      0.489|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.778|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.84/5|     4.77/5|
|**Pass Rate**      |   91.7%|       92.3%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.75/5|  4.85/5|  4.82/5|
|**Pass Rate**      |   88.2%|   93.2%|   94.1%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.75/5|  4.68/5|  4.81/5|  4.76/5|
|**Completeness**  |  4.85/5|  4.81/5|  4.89/5|  4.85/5|
|**Faithfulness**  |  4.87/5|  4.82/5|  4.89/5|  4.92/5|
|**Relevance**     |  4.96/5|  4.95/5|  4.97/5|  4.95/5|
|**Clarity**       |  4.98/5|  5.00/5|  4.99/5|  4.96/5|
|**Overall**       |  4.81/5|  4.75/5|  4.85/5|  4.82/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.75/5|      4.81/5|     4.69/5|
|**Completeness**  |  4.85/5|      4.89/5|     4.81/5|
|**Faithfulness**  |  4.87/5|      4.83/5|     4.91/5|
|**Relevance**     |  4.96/5|      4.93/5|     4.98/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.81/5|      4.84/5|     4.77/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.76s|        3.2%|
|**Total**       |    55.04s|        100%|

**Min Latency:** 2.88s  |  **Max Latency:** 108.05s

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
|**Single-Hop**  |    222|   205|       10|     7|      92.3%|
|**Multi-Hop**   |    236|   215|       19|     2|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   142|       13|     6|      88.2%|
|**Medium**  |    161|   150|       10|     1|      93.2%|
|**Hard**    |    136|   128|        6|     2|      94.1%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      90.9%|  1.000|         4.80/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.84/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.88/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.452|         4.68/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.570|         4.85/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.438|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  326.4s|
|**Questions/Second**   |    1.40|
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
|**Endpoint**      |  https://bfai-app-xcshqh7sqq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 13:59:49*
*Checkpoint: C064*
*Judge Model: gemini-3-flash-preview*